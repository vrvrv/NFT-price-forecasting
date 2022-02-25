from pytorch_forecasting import metrics, TemporalFusionTransformer


class temporal_fusion_transformer(TemporalFusionTransformer):
    def __init__(self, dataset, metric, metric_cfg, **kwargs):
        super().__init__(
            dataset=dataset,
            loss=getattr(metric, metrics)(**metric_cfg),
            **kwargs
        )
