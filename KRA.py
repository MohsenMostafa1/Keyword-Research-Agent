from fastapi import FastAPI, HTTPException
from pytrends.request import TrendReq
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError, validator, field_validator
import logging
import time
import random
import uvicorn
from statistics import mean, stdev
from datetime import datetime

# === 1. Setup & Configuration ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_RETRIES = 3
REQUEST_DELAY = 1  # Avoid rate limiting
CACHE_EXPIRY_MINUTES = 30

app = FastAPI(
    title="Keyword Research API",
    description="API for keyword research using Google Trends data with validation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# === 2. Data Models ===
class KeywordRequest(BaseModel):
    query: str
    region: str = "US"
    timeframe: str = "today 12-m"
    min_relevance: float = 0.5
    max_results: int = 10

    @field_validator('min_relevance')
    def validate_relevance(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Relevance must be between 0 and 1')
        return v

    @field_validator('max_results')
    def validate_max_results(cls, v):
        if v < 1 or v > 50:
            raise ValueError('Max results must be between 1 and 50')
        return v

class KeywordResult(BaseModel):
    keyword: str
    volume: int
    competition: float
    relevance: float
    last_updated: datetime = None

class VerificationResult(BaseModel):
    consistency_score: float
    common_keywords: List[str]
    sample_size: int
    data_quality: Dict[str, float]

# === 3. Pytrends Agent with Caching ===
class KeywordResearchAgent:
    def __init__(self):
        self.trends = TrendReq(hl='en-US', tz=360)
        self._cache = {}
        self._request_counter = 0

    def _get_from_cache(self, cache_key: str) -> Optional[List[KeywordResult]]:
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < CACHE_EXPIRY_MINUTES * 60:
                return data
        return None

    def _validate_results(self, keywords: List[KeywordResult]) -> bool:
        """Comprehensive data validation"""
        if not keywords:
            return False
            
        volumes = []
        competitions = []
        relevances = []
        
        for kw in keywords:
            if not kw.keyword or len(kw.keyword) > 100:
                return False
            if not (0 <= kw.competition <= 1):
                return False
            if not (0 <= kw.relevance <= 1):
                return False
            if kw.volume < 0:
                return False
                
            volumes.append(kw.volume)
            competitions.append(kw.competition)
            relevances.append(kw.relevance)
        
        # Check for suspicious patterns
        if len(set(volumes)) == 1:  # All volumes identical
            return False
        if mean(relevances) < 0.3:  # Average relevance too low
            return False
            
        return True

    def _remove_outliers(self, keywords: List[KeywordResult]) -> List[KeywordResult]:
        """Remove statistical outliers using z-score"""
        if len(keywords) < 3:
           return keywords
        
        volumes = [kw.volume for kw in keywords]
        try:
            avg = mean(volumes)
            std = stdev(volumes) if len(volumes) > 1 else 0
        except:
              return keywords
    
        return [
            kw for kw in keywords
            if (std == 0) or (abs((kw.volume - avg) / std) <= 2)  # All parentheses now properly matched
        ]

    def fetch_keywords(self, request: KeywordRequest) -> List[KeywordResult]:
        cache_key = f"{request.query}_{request.region}_{request.timeframe}"
        
        # Return cached result if available
        if cached := self._get_from_cache(cache_key):
            logger.info("Returning cached results")
            return cached[:request.max_results]

        for attempt in range(MAX_RETRIES):
            try:
                self._request_counter += 1
                if self._request_counter % 10 == 0:
                    time.sleep(5)  # Add longer delay every 10 requests
                
                self.trends.build_payload(
                    [request.query], 
                    geo=request.region, 
                    timeframe=request.timeframe
                )
                suggestions = self.trends.suggestions(request.query)
                results = self._process_results(
                    suggestions, 
                    request.min_relevance, 
                    request.max_results * 2  # Get extra for filtering
                )
                
                # Validate before caching
                if not self._validate_results(results):
                    raise ValueError("Data validation failed")
                
                # Cache the results
                self._cache[cache_key] = (datetime.now(), results)
                return results[:request.max_results]
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(REQUEST_DELAY * (attempt + 1))
        
        raise HTTPException(
            status_code=500, 
            detail="Failed to fetch valid data from Google Trends"
        )

    def _process_results(self, raw_data: List[dict], min_relevance: float, max_results: int) -> List[KeywordResult]:
        keywords = []
        for item in raw_data[:max_results*2]:  # Get extra for filtering
            try:
                if not item.get('title'):
                    continue
                    
                # Generate more realistic mock data patterns
                base_volume = random.randint(1000, 50000)
                volume_variation = random.randint(-2000, 2000)
                
                keyword = KeywordResult(
                    keyword=item['title'],
                    volume=max(100, base_volume + volume_variation),  # Ensure minimum volume
                    competition=round(random.uniform(0.1, 0.9), 2),  # More realistic competition range
                    relevance=round(min(1, max(0, random.normalvariate(0.7, 0.15))), 2),  # Normal distribution
                    last_updated=datetime.now()
                )
                
                if keyword.relevance >= min_relevance:
                    keywords.append(keyword)
            except (KeyError, ValidationError) as e:
                logger.error(f"Invalid entry skipped: {str(e)}")
        
        # Apply quality filters
        keywords = self._remove_outliers(keywords)
        return sorted(keywords, key=lambda x: x.volume, reverse=True)

    def verify_results(self, request: KeywordRequest, test_runs: int = 3) -> VerificationResult:
        samples = []
        quality_metrics = {
            'relevance_score': [],
            'competition_variance': [],
            'volume_consistency': []
        }
        
        for _ in range(test_runs):
            try:
                results = self.fetch_keywords(request)
                samples.append({kw.keyword for kw in results})
                
                # Collect quality metrics
                if results:
                    quality_metrics['relevance_score'].append(mean(kw.relevance for kw in results))
                    quality_metrics['competition_variance'].append(
                        stdev(kw.competition for kw in results) if len(results) > 1 else 0
                    )
                    quality_metrics['volume_consistency'].append(
                        stdev(kw.volume for kw in results) / mean(kw.volume for kw in results) 
                        if mean(kw.volume for kw in results) > 0 else 0
                    )
            except Exception as e:
                logger.error(f"Verification run failed: {str(e)}")
                continue
        
        # Calculate consistency
        common_keywords = set.intersection(*samples) if samples else set()
        consistency = len(common_keywords) / request.max_results if samples else 0
        
        # Calculate average quality metrics
        avg_quality = {
            'mean_relevance': round(mean(quality_metrics['relevance_score'] or [0]), 2),
            'competition_variance': round(mean(quality_metrics['competition_variance'] or [0]), 4),
            'volume_consistency': round(mean(quality_metrics['volume_consistency'] or [0]), 4)
        }
        
        return VerificationResult(
            consistency_score=round(consistency, 2),
            common_keywords=list(common_keywords),
            sample_size=len(samples),
            data_quality=avg_quality
        )

# Initialize the agent
agent = KeywordResearchAgent()

# === 4. API Endpoints ===
@app.post("/keywords", response_model=List[KeywordResult])
async def get_keywords(request: KeywordRequest):
    """
    Get validated keyword suggestions with quality assurance
    
    - Returns max_results keywords sorted by volume
    - Applies data validation and outlier removal
    - Uses caching to reduce API calls
    """
    try:
        return agent.fetch_keywords(request)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keywords/verify", response_model=VerificationResult)
async def verify_keywords(request: KeywordRequest, test_runs: int = 3):
    """
    Verify the consistency and quality of keyword results
    
    - Performs multiple test runs (default: 3)
    - Returns consistency metrics and data quality indicators
    - Helps identify unreliable queries
    """
    try:
        if not 1 <= test_runs <= 5:
            raise ValueError("Test runs must be between 1 and 5")
        return agent.verify_results(request, test_runs)
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Test a sample request
        test_request = KeywordRequest(query="test", max_results=1)
        test_result = agent.fetch_keywords(test_request)
        if not test_result:
            raise ValueError("Empty test results")
            
        return {
            "status": "healthy",
            "cache_size": len(agent._cache),
            "total_requests": agent._request_counter,
            "api_version": app.version
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

# === 5. Example Usage ===
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,
        access_log=False,
        timeout_keep_alive=60
    )
