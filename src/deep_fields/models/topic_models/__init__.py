from deep_fields.models.topic_models.dynamic import DynamicTopicEmbeddings, DynamicLDA, DynamicBinaryTopicEmbeddings
from deep_fields.models.topic_models.static import DiscreteLatentTopicNVI
from deep_fields.models.topic_models.supervised_static import RegressionDiscreteLatentTopicNVI, ClassificationDiscreteLatentTopicNVI, SupervisedTopicModel
from deep_fields.models.topic_models.supervised_dynamic import SupervisedDynamicTopicModel
from deep_fields.models.topic_models.baseline_models import NonSequentialClassifier, NonSequentialRegression, SequentialClassifier, SequentialRegression


class ModelFactory(object):
    _models: dict = {
        'discrete_latent_topic': DiscreteLatentTopicNVI,
        'dynamic_lda': DynamicLDA,
        'supervised_static_vanilla_topic': SupervisedTopicModel,
        'supervised_dynamic_topic_model': SupervisedDynamicTopicModel,
        'neural_dynamical_topic_embeddings': DynamicTopicEmbeddings,
        'binary_dynamical_topic_embeddings': DynamicBinaryTopicEmbeddings,
        'nonsequential_classifier': NonSequentialClassifier,
        'nonsequential_regressor': NonSequentialRegression,
        'sequential_classification': SequentialClassifier,
        'sequential_regression': SequentialRegression,
        'classification_miao': ClassificationDiscreteLatentTopicNVI,
        'regression_miao': RegressionDiscreteLatentTopicNVI
    }

    @classmethod
    def get_instance(cls, model_type: str, **kwargs):
        builder = cls._models.get(model_type)
        if not builder:
            raise ValueError(f'Unknown recognition model {model_type}')
        return builder(**kwargs)
