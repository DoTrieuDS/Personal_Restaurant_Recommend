from dependency_injector import containers, providers
from shared.settings import Settings
from modules.memory.short_term import SessionStore
from modules.recommendation.service import RecommendationService
from modules.recommendation.user_profile_service import UserProfileService
from modules.recommendation.feedback_learning import FeedbackLearningService
from modules.recommendation.monitoring import MetricsCollector, RecommendationLogger


class Container(containers.DeclarativeContainer):
    # 1. cấu hình
    config = providers.Singleton(Settings)

    # 2. infrastructure
    short_term_memory = providers.Singleton(
        SessionStore,
        settings=config,
    )
    
    # 3. monitoring và logging
    metrics_collector = providers.Singleton(
        MetricsCollector,
        memory=short_term_memory,
    )
    
    recommendation_logger = providers.Singleton(
        RecommendationLogger,
        log_level="INFO",
    )

    # 4. user profile và learning services
    user_profile_service = providers.Singleton(
        UserProfileService,
        memory=short_term_memory,
    )
    
    feedback_learning_service = providers.Singleton(
        FeedbackLearningService,
        memory=short_term_memory,
        user_profile_service=user_profile_service,
    )

    # 5. application services
    recommendation_service = providers.Factory(
        RecommendationService,
        memory=short_term_memory,
    )