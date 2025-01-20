from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import StandardScaler

class ScalerType(Enum):
    STANDARD_SCALER = StandardScaler
    MINMAX_SCALER = MinMaxScaler
    MAXABS_SCALER = MaxAbsScaler
    ROBUST_SCALER = RobustScaler
    QUANTILE_TRANSFORMER = QuantileTransformer