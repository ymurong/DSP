from src.transactions import transaction_app
from fastapi import FastAPI
import uvicorn
from fastapi_pagination import add_pagination
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title='Risk API',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transaction_app, prefix='/transactions', tags=['transactions'])

add_pagination(app)

if __name__ == '__main__':
    uvicorn.run('run:app', host='0.0.0.0', port=8000, reload=True, workers=1)
