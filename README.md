# 🚀 Stock Analysis Platform - Quick Start

Production-ready stock analysis API with AI predictions, technical analysis, and React frontend.

## ⚡ Quick Start (2 Steps)

### 1. Start Backend
```bash
start-backend.bat
```
📋 **Copy the API key** shown in the startup logs!

### 2. Start Frontend
Add API key to `frontend/.env`:
```
VITE_API_KEY=your_copied_key
```

Then run:
```bash
start-frontend.bat
```

**Open:** http://localhost:5173

---

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Complete setup instructions
- **[RUNNING_LOCALLY.md](RUNNING_LOCALLY.md)** - Detailed development guide
- **[DEPLOYMENT_VERIFICATION.md](DEPLOYMENT_VERIFICATION.md)** - Production deployment
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical overview

---

## 🎯 What's Setup

✅ **Backend (FastAPI)** - Port 8000
- API key authentication
- Per-IP rate limiting (60 req/min)
- CORS configured for frontend
- Comprehensive API endpoints
- AI-powered analysis

✅ **Frontend (React + Vite)** - Port 5173
- Modern React UI
- Automatic API authentication
- Real-time data visualization
- Hot module reloading

✅ **Security**
- API key authentication required
- Rate limiting middleware
- CORS protection
- Production-ready error handling

✅ **Testing**
- 70%+ code coverage target
- Security tests
- API endpoint tests
- Unit tests

---

## 🛠️ Manual Setup

If batch files don't work:

**Backend:**
```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

**Frontend:**
```powershell
cd frontend
npm install  # first time only
npm run dev
```

---

## 📊 API Endpoints

Access API documentation: http://localhost:8000/docs

Key endpoints:
- `/api/stock/{symbol}` - Stock data
- `/api/technical/{symbol}` - Technical indicators
- `/api/ai/analyze/{symbol}` - AI analysis
- `/api/quantitative/forecast/{symbol}` - ML predictions
- `/api/backtest-summary/{symbol}` - Strategy backtesting
- `/api/model-accuracy/{symbol}` - Model performance

---

## 🔧 Configuration

### Backend (`.env`)
```env
GEMINI_API_KEY=your_key
SERPER_API_KEY=your_key
ALLOWED_ORIGINS=http://localhost:5173
ENVIRONMENT=development
```

### Frontend (`frontend/.env`)
```env
VITE_API_KEY=backend_generated_key
```

---

## 🐳 Docker Deployment

```bash
# Setup secrets
mkdir secrets
echo "your_gemini_key" > secrets/gemini_api_key.txt
echo "your_serper_key" > secrets/serper_api_key.txt
echo "api_key_1,api_key_2" > secrets/api_keys.txt

# Start services
docker-compose up -d
```

---

## ❓ Troubleshooting

**"Authentication required" error:**
- Check `frontend/.env` has `VITE_API_KEY`
- Restart frontend after adding key

**"Connection refused":**
- Ensure backend is running on port 8000
- Check `http://localhost:8000/health`

**"Port already in use":**
```powershell
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

See [QUICK_START.md](QUICK_START.md) for detailed troubleshooting.

---

## 📈 Features

- **Real-time Analysis:** Live stock data and indicators
- **AI Predictions:** ML-powered price forecasting
- **Technical Analysis:** 20+ technical indicators
- **Fundamental Analysis:** Company metrics and valuation
- **Pattern Recognition:** Chart patterns and signals
- **Backtesting:** Strategy performance testing
- **Model Tracking:** Prediction accuracy monitoring
- **Sentiment Analysis:** News and market sentiment

---

## 🔬 Technology Stack

**Backend:**
- FastAPI (Python)
- TensorFlow/Keras (ML models)
- XGBoost, LSTM, GRU, Attention
- Google Gemini AI
- yFinance, pandas, numpy

**Frontend:**
- React 19
- Vite
- Plotly.js (charts)
- Styled Components

**Deployment:**
- Docker & Docker Compose
- Redis (rate limiting)
- Multi-stage builds
- Health checks

---

## 📝 License

MIT

---

## 🆘 Support

For issues or questions:
1. Check [QUICK_START.md](QUICK_START.md)
2. Review [RUNNING_LOCALLY.md](RUNNING_LOCALLY.md)
3. See API docs: http://localhost:8000/docs
