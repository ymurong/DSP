from src.transactions import transaction_app
from src.metadata import metadata_app
from fastapi import FastAPI, Request
import uvicorn
from fastapi_pagination import add_pagination
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(
    title='Risk API',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
)


@app.middleware('http')
async def add_process_time_header(request: Request, call_next):  # call_next将接收request请求做为参数
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers['X-Process-Time'] = str(process_time)  # 添加自定义的以“X-”开头的请求头
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        "http://localhost:4200"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transaction_app, prefix='/transactions', tags=['transactions'])
app.include_router(metadata_app, prefix='/metadata', tags=['metadata'])


add_pagination(app)

if __name__ == '__main__':
    uvicorn.run('run:app', host='0.0.0.0', port=8000, reload=True)
