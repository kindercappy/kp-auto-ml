class ModelPerformance():
    score = None
    model_name = None
    RMSE = None
    total_columns = None
    scaler_type = None
    selected_features = None
    num_of_clusters_if_used = None
    accuracy = None
    f1_score = None

    def __init__(self,score,model_name,RMSE = None,total_columns = 0,scaler_type = None,selected_features = None,num_of_clusters = None, accuracy = None, f1_score = None) -> None:
        self.model_name = model_name
        self.score = score
        self.RMSE = RMSE
        self.total_columns = total_columns
        self.scaler_type = scaler_type
        self.selected_features = selected_features
        self.num_of_clusters_if_used = num_of_clusters
        self.accuracy = accuracy
        self.f1_score = f1_score

    def to_dict(self):
        return vars(self)
 