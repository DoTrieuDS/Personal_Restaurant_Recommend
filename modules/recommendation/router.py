"""
FastAPI Router cho Travel Recommendation Module
Tích hợp BGE-M3 + FAISS + Llama-3 Pipeline
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import date, datetime
from .recommendation_pipeline import TravelRecommendationPipeline
from dependency_injector.wiring import inject, Provide
from shared.kernel import Container

# Initialize router
router = APIRouter(tags=["recommendation"])

class RecommendationRequest(BaseModel):
    """Request model cho recommendation"""
    user_query: str
    num_candidates: Optional[int] = 50
    num_results: Optional[int] = 10
    user_preferences: Optional[Dict] = None

class RecommendationResponse(BaseModel):
    """Response model cho recommendation"""
    user_query: str
    recommendations: List[Dict]
    pipeline_info: Dict
    success: bool
    message: str

class SearchRequest(BaseModel):
    """Request model cho search"""
    query: str
    top_k: Optional[int] = 10

@router.get("/health")
@inject
async def health_check(
    recommendation_service = Depends(Provide[Container.recommendation_service])
):
    """Health check endpoint"""
    try:
        # Initialize pipeline if not already
        recommendation_service.initialize_pipeline()
        
        # Get pipeline info
        info = recommendation_service.get_pipeline_info()
        return {
            "status": "healthy",
            "pipeline_version": info["pipeline_version"],
            "total_pois": info["embedding_service"]["vector_database"]["total_pois"],
            "device": info["embedding_service"]["model"]["device"],
            "service": "Recommendation Service"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/recommend", response_model=RecommendationResponse)
@inject
async def get_recommendations(
    request: RecommendationRequest,
    recommendation_service = Depends(Provide[Container.recommendation_service])
):
    """
    Main recommendation endpoint
    
    Args:
        request: RecommendationRequest với user query và preferences
        
    Returns:
        RecommendationResponse với POI recommendations
    """
    try:
        # Get recommendations từ service
        results = recommendation_service.get_recommendations(
            user_query=request.user_query,
            num_candidates=request.num_candidates,
            num_results=request.num_results
        )
        
        return RecommendationResponse(
            user_query=request.user_query,
            recommendations=results["recommendations"],
            pipeline_info=results["pipeline_info"],
            success=True,
            message=f"Found {len(results['recommendations'])} recommendations"
        )
        
    except Exception as e:
        logging.error(f"❌ Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@router.get("/recommend")
@inject
async def get_recommendations_get(
    query: str = Query(..., description="User query for POI recommendations"),
    num_candidates: int = Query(50, description="Number of candidates to retrieve"),
    num_results: int = Query(10, description="Number of final results to return"),
    recommendation_service = Depends(Provide[Container.recommendation_service])
):
    """
    GET endpoint cho recommendations (alternative)
    """
    request = RecommendationRequest(
        user_query=query,
        num_candidates=num_candidates,
        num_results=num_results
    )
    
    return await get_recommendations(request, recommendation_service)

@router.get("/pipeline/info")
@inject
async def get_pipeline_info(
    recommendation_service = Depends(Provide[Container.recommendation_service])
):
    """Lấy thông tin về pipeline components"""
    try:
        return recommendation_service.get_pipeline_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline info: {str(e)}")

@router.post("/search")
@inject
async def search_pois(
    request: SearchRequest,
    recommendation_service = Depends(Provide[Container.recommendation_service])
):
    """
    Direct search endpoint (chỉ BGE-M3 + FAISS, không reranking)
    """
    try:
        # Initialize pipeline if not already
        recommendation_service.initialize_pipeline()
        
        # Direct search qua embedding service
        results = recommendation_service.pipeline.embedding_service.search_similar_pois(
            query=request.query,
            k=request.top_k,
            return_embeddings=False
        )
        
        return {
            "query": request.query,
            "results": results["poi_results"],
            "search_info": results["search_info"],
            "success": True
        }
        
    except Exception as e:
        logging.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")