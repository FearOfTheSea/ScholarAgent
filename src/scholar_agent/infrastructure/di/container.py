"""Dependency-injector configuration for ScholarAgent."""

from dependency_injector import containers, providers

from scholar_agent.application.services.generation_count_policy import (
    GenerationCountPolicy,
)
from scholar_agent.application.services.mission_insights import MissionInsightsService
from scholar_agent.application.services.mission_planning import MissionPlanner
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.services.tutor_turn_service import TutorTurnService
from scholar_agent.application.use_cases.advance_study_session import (
    AdvanceStudySessionUseCase,
)
from scholar_agent.application.use_cases.answer_question import AnswerQuestionUseCase
from scholar_agent.application.use_cases.ask_study_agent import AskStudyAgentUseCase
from scholar_agent.application.use_cases.assess_learner_response import (
    AssessLearnerResponseUseCase,
)
from scholar_agent.application.use_cases.build_document_brief import (
    BuildDocumentBriefUseCase,
)
from scholar_agent.application.use_cases.check_runtime_readiness import (
    CheckRuntimeReadinessUseCase,
)
from scholar_agent.application.use_cases.complete_study_session import (
    CompleteStudySessionUseCase,
)
from scholar_agent.application.use_cases.continue_study_session import (
    ContinueStudySessionUseCase,
)
from scholar_agent.application.use_cases.delete_document import DeleteDocumentUseCase
from scholar_agent.application.use_cases.delete_study_session import (
    DeleteStudySessionUseCase,
)
from scholar_agent.application.use_cases.explain_concept import ExplainConceptUseCase
from scholar_agent.application.use_cases.export_mission_record import (
    ExportMissionRecordUseCase,
)
from scholar_agent.application.use_cases.generate_flashcards import (
    GenerateFlashcardsUseCase,
)
from scholar_agent.application.use_cases.generate_quiz import GenerateQuizUseCase
from scholar_agent.application.use_cases.get_mission_insights import (
    GetMissionInsightsUseCase,
)
from scholar_agent.application.use_cases.get_study_session import GetStudySessionUseCase
from scholar_agent.application.use_cases.ingest_document import IngestDocumentUseCase
from scholar_agent.application.use_cases.list_documents import ListDocumentsUseCase
from scholar_agent.application.use_cases.list_study_sessions import (
    ListStudySessionsUseCase,
)
from scholar_agent.application.use_cases.start_study_session import (
    StartStudySessionUseCase,
)
from scholar_agent.application.use_cases.summarize_document import (
    SummarizeDocumentUseCase,
)
from scholar_agent.application.use_cases.verify_mission_ledger import (
    VerifyMissionLedgerUseCase,
)
from scholar_agent.config.settings import Settings
from scholar_agent.infrastructure.adapters.faiss_repository import FAISSRepository
from scholar_agent.infrastructure.adapters.in_memory_store import InMemoryStore
from scholar_agent.infrastructure.adapters.langchain_retriever import LangChainRetriever
from scholar_agent.infrastructure.adapters.langchain_text_chunker import (
    LangChainTextChunker,
)
from scholar_agent.infrastructure.adapters.langgraph_mission_runner import (
    LangGraphMissionRunner,
)
from scholar_agent.infrastructure.adapters.langgraph_runner import LangGraphRunner
from scholar_agent.infrastructure.adapters.langgraph_tutor_runner import (
    LangGraphTutorRunner,
)
from scholar_agent.infrastructure.adapters.local_document_library import (
    LocalDocumentLibrary,
)
from scholar_agent.infrastructure.adapters.ollama_adapter import OllamaAdapter
from scholar_agent.infrastructure.adapters.pymupdf_loader import PyMuPDFLoader
from scholar_agent.infrastructure.adapters.quick_ask_runner import QuickAskRunner
from scholar_agent.infrastructure.adapters.scratch_gpt.scratch_gpt_adapter import (
    ScratchGPTAdapter,
)
from scholar_agent.infrastructure.adapters.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from scholar_agent.infrastructure.adapters.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
)
from scholar_agent.infrastructure.tools.answer_question_tool import AnswerQuestionTool
from scholar_agent.infrastructure.tools.assess_learner_response_tool import (
    AssessLearnerResponseTool,
)
from scholar_agent.infrastructure.tools.build_document_map_tool import (
    BuildDocumentMapTool,
)
from scholar_agent.infrastructure.tools.capabilities import (
    MISSION_CAPABILITIES,
    STUDY_CAPABILITIES,
)
from scholar_agent.infrastructure.tools.citation_lookup_tool import CitationLookupTool
from scholar_agent.infrastructure.tools.explain_concept_tool import ExplainConceptTool
from scholar_agent.infrastructure.tools.generate_flashcards_tool import (
    GenerateFlashcardsTool,
)
from scholar_agent.infrastructure.tools.generate_quiz_tool import GenerateQuizTool
from scholar_agent.infrastructure.tools.semantic_search_tool import SemanticSearchTool
from scholar_agent.infrastructure.tools.study_tool_executor import StudyToolExecutor
from scholar_agent.infrastructure.tools.summarize_document_tool import (
    SummarizeDocumentTool,
)


class Container(containers.DeclarativeContainer):
    """Maps application ports to the local infrastructure implementations."""

    config = providers.Configuration()

    validation_service = providers.Singleton(RequestValidationService)
    generation_count_policy = providers.Singleton(GenerationCountPolicy)
    llm_provider = providers.Selector(
        config.llm_provider_type,
        ollama=providers.Singleton(
            OllamaAdapter,
            model_name=config.model_name,
            base_url=config.ollama_url,
            context_length=config.llm_context_length,
            maximum_tokens=config.llm_max_tokens,
        ),
        scratch_gpt=providers.Singleton(
            ScratchGPTAdapter,
            context_length=config.llm_context_length,
            maximum_tokens=config.llm_max_tokens,
            checkpoint_path=config.scratch_gpt_checkpoint_path,
        ),
    )
    embedding_provider = providers.Singleton(
        SentenceTransformerEmbedding,
        model_name=config.embedding_model_name,
        device=config.embedding_device,
    )
    vector_store = providers.Singleton(
        FAISSRepository,
        database_path=config.vector_db_path,
    )
    retriever = providers.Singleton(
        LangChainRetriever,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_limit=config.retrieval_top_k,
    )
    pdf_loader = providers.Singleton(PyMuPDFLoader)
    text_chunker = providers.Singleton(
        LangChainTextChunker,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    document_library = providers.Singleton(
        LocalDocumentLibrary,
        library_path=config.document_library_path,
    )
    document_repository = providers.Singleton(
        SQLiteDocumentRepository,
        database_path=config.catalog_db_path,
    )
    study_session_repository = providers.Singleton(
        SQLiteStudySessionRepository,
        database_path=config.catalog_db_path,
    )
    mission_state_service = providers.Factory(
        MissionStateService,
        session_repository=study_session_repository,
    )
    memory_store = providers.Singleton(InMemoryStore)
    mission_planner = providers.Factory(
        MissionPlanner,
        llm_provider=llm_provider,
        maximum_objectives=config.agent_max_objectives,
    )

    check_runtime_readiness_use_case = providers.Factory(
        CheckRuntimeReadinessUseCase,
        llm_provider=llm_provider,
    )
    ingest_document_use_case = providers.Factory(
        IngestDocumentUseCase,
        document_repository=document_repository,
        document_library=document_library,
        pdf_loader=pdf_loader,
        text_chunker=text_chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        validation_service=validation_service,
        maximum_upload_bytes=config.maximum_upload_bytes,
    )
    list_documents_use_case = providers.Factory(
        ListDocumentsUseCase,
        document_repository=document_repository,
    )
    delete_document_use_case = providers.Factory(
        DeleteDocumentUseCase,
        document_repository=document_repository,
        document_library=document_library,
        vector_store=vector_store,
        session_repository=study_session_repository,
    )
    answer_question_use_case = providers.Factory(
        AnswerQuestionUseCase,
        llm_provider=llm_provider,
        retriever=retriever,
        validation_service=validation_service,
    )
    summarize_document_use_case = providers.Factory(
        SummarizeDocumentUseCase,
        llm_provider=llm_provider,
        vector_store=vector_store,
    )
    generate_quiz_use_case = providers.Factory(
        GenerateQuizUseCase,
        llm_provider=llm_provider,
        vector_store=vector_store,
        count_policy=generation_count_policy,
    )
    generate_flashcards_use_case = providers.Factory(
        GenerateFlashcardsUseCase,
        llm_provider=llm_provider,
        vector_store=vector_store,
        count_policy=generation_count_policy,
    )

    semantic_search_tool = providers.Factory(SemanticSearchTool, retriever=retriever)
    answer_question_tool = providers.Factory(
        AnswerQuestionTool,
        use_case=answer_question_use_case,
    )
    summarize_document_tool = providers.Factory(
        SummarizeDocumentTool,
        use_case=summarize_document_use_case,
    )
    generate_quiz_tool = providers.Factory(
        GenerateQuizTool,
        use_case=generate_quiz_use_case,
    )
    generate_flashcards_tool = providers.Factory(
        GenerateFlashcardsTool,
        use_case=generate_flashcards_use_case,
    )
    citation_lookup_tool = providers.Factory(
        CitationLookupTool,
        vector_store=vector_store,
    )
    tool_executor = providers.Singleton(
        StudyToolExecutor,
        tools=providers.Dict(
            semantic_search=semantic_search_tool,
            answer_question=answer_question_tool,
            summarize_document=summarize_document_tool,
            generate_quiz=generate_quiz_tool,
            generate_flashcards=generate_flashcards_tool,
            citation_lookup=citation_lookup_tool,
        ),
        capabilities=providers.Object(STUDY_CAPABILITIES),
    )
    graph_runner = providers.Singleton(LangGraphRunner, tool_executor=tool_executor)
    build_document_brief_use_case = providers.Factory(
        BuildDocumentBriefUseCase,
        llm_provider=llm_provider,
        vector_store=vector_store,
        session_repository=study_session_repository,
    )
    explain_concept_use_case = providers.Factory(
        ExplainConceptUseCase,
        llm_provider=llm_provider,
        vector_store=vector_store,
        session_repository=study_session_repository,
    )
    assess_learner_response_use_case = providers.Factory(
        AssessLearnerResponseUseCase,
        llm_provider=llm_provider,
        vector_store=vector_store,
        session_repository=study_session_repository,
    )
    build_document_map_tool = providers.Factory(
        BuildDocumentMapTool,
        use_case=build_document_brief_use_case,
    )
    explain_concept_tool = providers.Factory(
        ExplainConceptTool,
        use_case=explain_concept_use_case,
    )
    assess_learner_response_tool = providers.Factory(
        AssessLearnerResponseTool,
        use_case=assess_learner_response_use_case,
    )
    mission_tool_executor = providers.Singleton(
        StudyToolExecutor,
        tools=providers.Dict(
            semantic_search=semantic_search_tool,
            summarize_document=summarize_document_tool,
            generate_quiz=generate_quiz_tool,
            generate_flashcards=generate_flashcards_tool,
            citation_lookup=citation_lookup_tool,
            build_document_map=build_document_map_tool,
            explain_concept=explain_concept_tool,
            assess_learner_response=assess_learner_response_tool,
        ),
        capabilities=providers.Object(MISSION_CAPABILITIES),
    )
    agent_runner = providers.Singleton(
        QuickAskRunner, tool_executor=mission_tool_executor
    )
    ask_study_agent_use_case = providers.Factory(
        AskStudyAgentUseCase,
        agent_runner=agent_runner,
        validation_service=validation_service,
    )
    start_study_session_use_case = providers.Factory(
        StartStudySessionUseCase,
        brief_use_case=build_document_brief_use_case,
        session_repository=study_session_repository,
        validation_service=validation_service,
        mission_planner=mission_planner,
        state_service=mission_state_service,
    )
    tutor_turn_service = providers.Factory(
        TutorTurnService,
        llm_provider=llm_provider,
        retriever=retriever,
        session_repository=study_session_repository,
        state_service=mission_state_service,
    )
    tutor_runner = providers.Factory(
        LangGraphTutorRunner,
        turn_service=tutor_turn_service,
    )
    continue_study_session_use_case = providers.Factory(
        ContinueStudySessionUseCase,
        tutor_runner=tutor_runner,
        validation_service=validation_service,
    )
    get_study_session_use_case = providers.Factory(
        GetStudySessionUseCase,
        session_repository=study_session_repository,
    )
    delete_study_session_use_case = providers.Factory(
        DeleteStudySessionUseCase,
        session_repository=study_session_repository,
    )
    mission_runner = providers.Factory(
        LangGraphMissionRunner,
        tool_executor=mission_tool_executor,
        session_repository=study_session_repository,
        maximum_actions_per_turn=config.agent_max_actions_per_turn,
        maximum_actions_per_session=config.agent_max_actions_per_session,
        state_service=mission_state_service,
    )
    advance_study_session_use_case = providers.Factory(
        AdvanceStudySessionUseCase,
        mission_runner=mission_runner,
        validation_service=validation_service,
    )
    complete_study_session_use_case = providers.Factory(
        CompleteStudySessionUseCase,
        session_repository=study_session_repository,
        state_service=mission_state_service,
    )
    list_study_sessions_use_case = providers.Factory(
        ListStudySessionsUseCase,
        session_repository=study_session_repository,
    )
    mission_insights_use_case = providers.Factory(
        GetMissionInsightsUseCase,
        session_repository=study_session_repository,
        insights_service=providers.Factory(
            MissionInsightsService,
            maximum_actions_per_session=config.agent_max_actions_per_session,
        ),
    )
    verify_mission_ledger_use_case = providers.Factory(
        VerifyMissionLedgerUseCase,
        session_repository=study_session_repository,
    )
    export_mission_record_use_case = providers.Factory(
        ExportMissionRecordUseCase,
        session_repository=study_session_repository,
        insights_use_case=mission_insights_use_case,
    )


def build_container(settings: Settings) -> Container:
    """Create a configured local application container."""
    container = Container()
    container.config.from_dict(settings.to_container_config())
    return container
