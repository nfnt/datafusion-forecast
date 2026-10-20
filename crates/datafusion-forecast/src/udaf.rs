//! The `forecast` aggregate and the `forecast_quantile_levels` scalar function.

use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use datafusion::{
    arrow::datatypes::{DataType, Field, FieldRef, Fields},
    common::{Result, ScalarValue, exec_err, not_impl_err, plan_err},
    logical_expr::{
        Accumulator, AggregateUDFImpl, ColumnarValue, GroupsAccumulator, ScalarFunctionArgs,
        ScalarUDFImpl, Signature, Volatility,
        function::{AccumulatorArgs, StateFieldsArgs},
        utils::format_state_name,
    },
    physical_expr::expressions::Literal,
};

use crate::{
    accumulator::{ForecastAccumulator, ForecastGroupsAccumulator, GroupForecaster, Layout},
    convert::{TS_FIELD, output_type},
    forecaster::Forecaster,
};

/// `forecast(ts, targets, horizon [, past [, known]])`: a forecast of each group by a
/// [`Forecaster`].
///
/// - `ts`: timestamp, date or integer time of each row. Rows are sorted by it and must lie
///   on a regular grid whose step is the group's smallest gap; grid slots without a row are
///   missing values. Calendar months need an integer month number instead.
/// - `targets`: a struct of numeric columns, e.g. `named_struct('sales', sales)`, whose
///   fields are forecast jointly. NULL or NaN is a missing value.
/// - `horizon`: number of steps to forecast, an integer literal.
/// - `past`: optional struct of numeric covariates known over the context only.
/// - `known`: optional struct of numeric covariates known over the context and the horizon.
///   The last `horizon` rows of each group are then the horizon and their targets must be
///   NULL. Pass `NULL` for `past` to give only known covariates.
///
/// Returns a list of `horizon` steps
/// `{ts, <target>: {median, quantiles}, ...}`, quantiles at the levels of
/// `forecast_quantile_levels()`. Groups with fewer than two rows or no observed target
/// value are NULL.
#[derive(Clone)]
pub struct ForecastUdaf {
    name: String,
    signature: Signature,
    forecaster: Arc<dyn Forecaster>,
    batch_size: usize,
}

impl ForecastUdaf {
    pub fn new(forecaster: Arc<dyn Forecaster>) -> Self {
        Self {
            name: "forecast".into(),
            signature: Signature::user_defined(Volatility::Immutable),
            forecaster,
            batch_size: 64,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Most series per forecaster call (default 64). Only groups whose context has the
    /// same length are batched together.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn forecaster(&self) -> &Arc<dyn Forecaster> {
        &self.forecaster
    }

    fn create(&self, args: &AccumulatorArgs) -> Result<GroupForecaster> {
        if args.is_distinct {
            return not_impl_err!("{}(DISTINCT ...) is not supported", self.name);
        }
        let max_horizon = self.forecaster.max_horizon();
        let horizon = match args.exprs[2].downcast_ref::<Literal>().map(Literal::value) {
            Some(ScalarValue::Int64(Some(h)))
                if *h >= 1 && usize::try_from(*h).is_ok_and(|h| h <= max_horizon) =>
            {
                *h as usize
            }
            Some(v) => {
                return exec_err!(
                    "{}: horizon must be between 1 and {max_horizon}, got {v}",
                    self.name
                );
            }
            None => return plan_err!("{}: horizon must be a literal", self.name),
        };
        let q = self.forecaster.quantile_levels().len();
        let median = self.forecaster.median_index();
        if median >= q {
            return exec_err!(
                "{}: forecaster has median index {median} but {q} quantile levels",
                self.name
            );
        }
        let width = |i: usize| {
            args.expr_fields.get(i).map_or(0, |f| match f.data_type() {
                DataType::Struct(fields) => fields.len(),
                _ => 0,
            })
        };
        let DataType::Struct(targets) = args.expr_fields[1].data_type() else {
            return plan_err!("{}: targets must be a struct", self.name);
        };
        if targets.is_empty() {
            return plan_err!("{}: targets must have at least one field", self.name);
        }
        Ok(GroupForecaster {
            forecaster: self.forecaster.clone(),
            batch_size: self.batch_size,
            layout: Layout {
                variates: targets.len(),
                past: width(3),
                known: width(4),
                horizon,
            },
            ts_type: args.expr_fields[0].data_type().clone(),
            variates: targets.iter().map(|f| f.name().clone()).collect(),
        })
    }
}

impl std::fmt::Debug for ForecastUdaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForecastUdaf")
            .field("name", &self.name)
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

impl PartialEq for ForecastUdaf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && std::ptr::addr_eq(
                Arc::as_ptr(&self.forecaster),
                Arc::as_ptr(&other.forecaster),
            )
            && self.batch_size == other.batch_size
    }
}

impl Eq for ForecastUdaf {}

impl Hash for ForecastUdaf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        Arc::as_ptr(&self.forecaster).cast::<()>().hash(state);
        self.batch_size.hash(state);
    }
}

/// Numeric struct fields become Float32.
fn coerce_struct(name: &str, arg: &str, data_type: &DataType) -> Result<DataType> {
    let DataType::Struct(fields) = data_type else {
        return plan_err!(
            "{name}: {arg} must be a struct such as named_struct('x', x), got {data_type}"
        );
    };
    let fields: Result<Fields> = fields
        .iter()
        .map(|f| {
            if f.data_type().is_numeric() || f.data_type().is_null() {
                Ok(Field::new(f.name(), DataType::Float32, true))
            } else {
                plan_err!(
                    "{name}: {arg} field {} must be numeric, got {}",
                    f.name(),
                    f.data_type()
                )
            }
        })
        .collect();
    Ok(DataType::Struct(fields?))
}

impl AggregateUDFImpl for ForecastUdaf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let name = &self.name;
        if !(3..=5).contains(&arg_types.len()) {
            return plan_err!(
                "{name} takes (ts, targets, horizon [, past [, known]]), got {} arguments",
                arg_types.len()
            );
        }
        let ts = match &arg_types[0] {
            t @ (DataType::Timestamp(..) | DataType::Date32 | DataType::Date64) => t.clone(),
            t if t.is_integer() => DataType::Int64,
            t => return plan_err!("{name}: ts must be a timestamp, date or integer, got {t}"),
        };
        if !arg_types[2].is_integer() {
            return plan_err!("{name}: horizon must be an integer, got {}", arg_types[2]);
        }
        let mut coerced = vec![
            ts,
            coerce_struct(name, "targets", &arg_types[1])?,
            DataType::Int64,
        ];
        for (arg, t) in ["past", "known"].iter().zip(&arg_types[3..]) {
            coerced.push(if t.is_null() {
                DataType::Null
            } else {
                coerce_struct(name, arg, t)?
            });
        }
        Ok(coerced)
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let DataType::Struct(targets) = &arg_types[1] else {
            return plan_err!("{}: targets must be a struct", self.name);
        };
        if targets.is_empty() {
            return plan_err!("{}: targets must have at least one field", self.name);
        }
        let variates: Vec<String> = targets.iter().map(|f| f.name().clone()).collect();
        if variates.iter().any(|v| v == TS_FIELD) {
            return plan_err!("{}: a target cannot be named {TS_FIELD:?}", self.name);
        }
        Ok(output_type(
            &arg_types[0],
            &variates,
            self.forecaster.quantile_levels().len(),
        ))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ForecastAccumulator::new(self.create(&acc_args)?)))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let list = |state: &str, item: DataType| -> FieldRef {
            Arc::new(Field::new_list(
                format_state_name(args.name, state),
                Field::new_list_field(item, true),
                true,
            ))
        };
        Ok(vec![
            list("ts", DataType::Int64),
            list("targets", DataType::Float32),
            list("past", DataType::Float32),
            list("known", DataType::Float32),
        ])
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(ForecastGroupsAccumulator::new(
            self.create(&args)?,
        )))
    }
}

/// `forecast_quantile_levels()`: the probability levels of the `quantiles` lists returned
/// by `forecast`.
#[derive(Debug, Clone)]
pub struct QuantileLevelsUdf {
    name: String,
    signature: Signature,
    levels: Vec<f64>,
}

impl QuantileLevelsUdf {
    pub fn new(forecaster: &dyn Forecaster) -> Self {
        Self {
            name: "forecast_quantile_levels".into(),
            signature: Signature::nullary(Volatility::Immutable),
            levels: forecaster.quantile_levels().to_vec(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl PartialEq for QuantileLevelsUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.levels.len() == other.levels.len()
            && self
                .levels
                .iter()
                .zip(&other.levels)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for QuantileLevelsUdf {}

impl Hash for QuantileLevelsUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.levels.iter().for_each(|l| l.to_bits().hash(state));
    }
}

impl ScalarUDFImpl for QuantileLevelsUdf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::new_list(DataType::Float64, true))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let levels: Vec<ScalarValue> = self.levels.iter().map(|&l| ScalarValue::from(l)).collect();
        Ok(ColumnarValue::Scalar(ScalarValue::List(
            ScalarValue::new_list_nullable(&levels, &DataType::Float64),
        )))
    }
}
