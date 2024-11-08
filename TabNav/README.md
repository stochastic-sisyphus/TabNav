# TabNav

TabNav is an AI-powered browser extension that helps you organize and analyze your browser tabs using advanced machine learning techniques.

## Features
- Smart tab organization using AI clustering
- Content analysis and summarization
- Multi-language support
- Interactive analytics dashboard
- Import/export functionality

## Project Structure
```
tabnav/
├── backend/                 # Python backend service
│   ├── api/                # FastAPI application
│   ├── models/             # Data models
│   ├── services/           # Core services
│   └── utils/              # Utilities
├── extension/              # Chrome extension
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── store/        # State management
│   │   └── utils/        # Utilities
│   └── public/           # Static assets
└── docs/                 # Documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Redis server
- Chrome browser

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/tabnav.git
cd tabnav
```

2. Install dependencies:
```bash
# Backend
./scripts/install.sh

# Extension
cd extension
npm install
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Start services:
```bash
# Start backend
docker-compose up -d

# Build extension
cd extension
npm run build
```

5. Load extension in Chrome:
- Open chrome://extensions/
- Enable Developer Mode
- Click "Load unpacked" and select extension/build

## Development
See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development guide.

## Architecture
See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture details.

## Contributing
1. Fork repository
2. Create feature branch
3. Submit pull request

## License
MIT License

## Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd extension
npm test
```

## API Documentation
After starting the backend server, visit:
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Security
Please report security vulnerabilities to security@yourdomain.com

## Support
For support, please open an issue or contact support@yourdomain.com