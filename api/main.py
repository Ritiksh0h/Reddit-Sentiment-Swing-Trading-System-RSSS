"""
RSSS FastAPI application — entry point.
Routes are in api/routes/*.py
Run: uvicorn api.main:app --reload --port 8000
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.routes.health      import router as health_router
from api.routes.portfolio   import router as portfolio_router
from api.routes.predictions import router as predictions_router
from api.routes.performance import router as performance_router
from api.routes.research    import router as research_router

app = FastAPI(title='RSSS Paper Trading API', version='3.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health_router)
app.include_router(portfolio_router)
app.include_router(predictions_router)
app.include_router(performance_router)
app.include_router(research_router)


@app.get('/dashboard')
def serve_dashboard():
    dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'index.html'
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail='dashboard/index.html not found')
    return FileResponse(str(dashboard_path))
