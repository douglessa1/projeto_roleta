import os
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import timedelta, datetime
from typing import Optional, List

from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv

# ---------------------------------------------------------
# LOAD ENV
# ---------------------------------------------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 600

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Config bcrypt/passlib
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------
# FASTAPI INIT
# ---------------------------------------------------------
app = FastAPI()

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# STATIC FILES FRONTEND
# ---------------------------------------------------------
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------
# JWT HELPERS
# ---------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

def verify_password(plain, hashed):
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def authenticate_user(username: str, password: str):
    if username != ADMIN_USERNAME:
        return False
    if password != ADMIN_PASSWORD:
        return False
    return True

# ---------------------------------------------------------
# LOGIN ENDPOINT
# ---------------------------------------------------------
@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not await authenticate_user(form_data.username, form_data.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}

# ---------------------------------------------------------
# TOKEN VERIFIER
# ---------------------------------------------------------
async def get_current_user(Authorization: str = Header(None)):
    if Authorization is None:
        raise HTTPException(status_code=401, detail="Token missing")

    token = Authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = payload.get("sub")
        if user is None:
            raise HTTPException(status_code=401)
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------------------------------------------------------
# DASHBOARD DATA MOCK
# (Você pode futuramente integrar Supabase)
# ---------------------------------------------------------
bot_status = "PAUSED"
saldo_simulado = 1000.00
p_l = 0.0
roi = 0.0
performance_history = {"labels": [], "data": []}

LOG_BUFFER: List[str] = []

def log(msg: str):
    global LOG_BUFFER
    if len(LOG_BUFFER) > 200:
        LOG_BUFFER = LOG_BUFFER[-200:]
    LOG_BUFFER.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---------------------------------------------------------
@app.get("/api/kpis")
async def get_kpis(user: str = Depends(get_current_user)):
    return {
        "saldo_simulado": saldo_simulado,
        "p_l": p_l,
        "roi": roi,
        "bot_status": bot_status,
    }

# ---------------------------------------------------------
@app.post("/api/bot/toggle")
async def toggle_bot(data: dict, user: str = Depends(get_current_user)):
    global bot_status
    bot_status = data["status"]
    log(f"BOT ALTERADO PARA: {bot_status}")
    return {"status": bot_status}

# ---------------------------------------------------------
@app.get("/api/performance-history")
async def perf_history(user: str = Depends(get_current_user)):
    return performance_history

# ---------------------------------------------------------
@app.get("/api/ai/gemini-analysis")
async def ai_analysis(user: str = Depends(get_current_user)):
    # MOCK — substitua quando for conectar ao Gemini
    return {
        "status_code": 200,
        "analysis": "Nenhuma análise disponível no momento (mock)."
    }

# ---------------------------------------------------------
# LOGS FRONTEND
# ---------------------------------------------------------
@app.get("/api/ui/logs")
async def ui_logs(user: str = Depends(get_current_user)):
    return {"logs": LOG_BUFFER[-100:]}

# ---------------------------------------------------------
# FRONTEND CONFIG
# ---------------------------------------------------------
@app.get("/api/config/frontend")
async def frontend_cfg():
    return {
        "API_RUNNING": True,
        "SPIN_INTERVAL_SECONDS": 15,
        "HOST": os.getenv("HOST"),
        "PORT": os.getenv("PORT")
    }

# ---------------------------------------------------------
# SERVE INDEX
# ---------------------------------------------------------
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# ---------------------------------------------------------
# AUTO START SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print("\n======================================")
    print("🚀 FASTAPI + ROLETABOT SERVER RODANDO")
    print(f"🌐 URL: http://{host}:{port}")
    print("======================================\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
