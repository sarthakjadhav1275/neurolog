# 🚀 NeuroLog - Advanced Log Analysis Platform

<div align="center">
  <img src="https://img.shields.io/badge/Streamlit-1.28.0+-blue.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3.8+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg" alt="Status">
</div>

## 📖 Description

**NeuroLog** is an enterprise-grade log analysis platform that transforms raw log files into actionable insights using advanced AI-powered analytics. With a modern, interactive interface and powerful machine learning capabilities, NeuroLog helps developers and DevOps teams quickly identify issues, patterns, and trends in their system logs.

## ✨ Key Features

### 🎯 Core Functionality
- **📁 Smart Log Parsing** - Support for multiple log formats with batch processing
- **🔍 Anomaly Detection** - Rule-based and ML-powered anomaly identification
- **🧩 Pattern Discovery** - Recurring failure pattern analysis
- **📊 Interactive Visualizations** - Beautiful charts and real-time metrics
- **📄 Enhanced Exports** - PDF, JSON, CSV, and Markdown reports

### 🌟 Advanced Features
- **🎨 Theme System** - Smooth dark/light mode toggle with animations
- **📡 Real-time Streaming** - Live log monitoring with instant analysis
- **🔮 Time-series Forecasting** - ML-based error trend prediction
- **⚡ Severity Scoring** - Advanced anomaly severity analysis
- **🔌 API Integration** - RESTful endpoints for programmatic access
- **🚀 Performance Optimization** - Batch processing for large files

### 🎨 Enhanced UI/UX
- **Modern Design** - Clean, professional interface with smooth animations
- **Interactive Elements** - Hover effects, transitions, and micro-interactions
- **Responsive Layout** - Works perfectly on all screen sizes
- **Accessibility** - Semantic HTML and keyboard navigation support

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.28+
- **Backend**: Python 3.8+
- **Machine Learning**: scikit-learn, Isolation Forest, TF-IDF
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, interactive charts
- **Export**: ReportLab (PDF), JSON, CSV
- **API**: FastAPI, Uvicorn
- **Database**: SQLite for user management

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/neurolog.git
   cd neurolog
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app_stable.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Docker Setup (Optional)

```bash
# Build the Docker image
docker build -t neurolog .

# Run the container
docker run -p 8501:8501 neurolog
```

## 📖 Usage Guide

### 1. **Upload Logs**
- Drag and drop your `.log` or `.txt` files
- Supports multiple log formats
- Automatic batch processing for large files

### 2. **Configure Analysis**
- Set anomaly detection parameters
- Choose between rule-based and ML detection
- Adjust contamination levels for sensitivity

### 3. **Explore Results**
- **Overview**: Structured log view with filters
- **Anomalies**: Detected issues with severity scores
- **Clusters**: Grouped similar failure patterns
- **Patterns**: Recurring issues and trends
- **Forecast**: Predictive error analysis
- **Live Stream**: Real-time log monitoring

### 4. **Export Insights**
- Download comprehensive reports in multiple formats
- Generate PDF documentation
- Export data for further analysis

## 🎯 Use Cases

### � Development Teams
- Debug application issues faster
- Monitor error patterns in development
- Improve code quality through log analysis

### 🏢 DevOps & IT Operations
- Monitor system health in real-time
- Identify performance bottlenecks
- Automate incident response workflows

### 📊 Analytics & Monitoring
- Track system metrics over time
- Generate compliance reports
- Data-driven decision making

## 🏗️ Architecture

```
NeuroLog/
├── app_stable.py          # Enhanced production version
├── app.py                 # Original application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── anomaly.py            # ML anomaly detection
├── clustering.py         # Failure pattern clustering
├── parser.py             # Log parsing utilities
├── assets/               # Static assets (logos, images)
├── neurolog_users.sqlite3 # User database (auto-created)
└── labelled_logs_example.csv # Sample labeled data
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom database path
export DB_PATH="/path/to/your/database.sqlite3"

# Optional: Set custom assets directory
export ASSETS_DIR="/path/to/assets"
```

### Log Format Support
- **Strict Format**: `YYYY-MM-DD HH:MM:SS LEVEL message`
- **Flexible Format**: Timestamp anywhere with level keywords
- **Custom Formats**: Extend parser.py for additional formats

## 📊 API Documentation

### Health Check
```http
GET /api/health
```

### Get Statistics
```http
GET /api/stats
```

### Authentication
All API endpoints support JWT-based authentication for enterprise deployments.

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Test with sample data
streamlit run app_stable.py -- --test-mode
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 📊 Forecasting Error
**Issue**: `Invalid frequency: H. Failed to parse...`
**Solution**: This has been fixed in the latest version. Ensure you're using `app_stable.py`

#### 🎨 Theme Toggle Not Working
**Solution**: Refresh the page and try again. Theme preferences are saved in session state.

#### 📡 Live Stream Issues
**Solution**: Ensure your log file is being updated in real-time. The feature monitors file changes.

#### 📄 PDF Export Fails
**Solution**: Install required dependencies:
```bash
pip install reportlab
```

#### 🚀 Performance Issues
**Solution**: For large files (>10MB), the app automatically uses batch processing. Consider filtering your data first.

### System Requirements
- **Python**: 3.8+
- **Memory**: 4GB+ recommended for large files
- **Storage**: 100MB+ for application and data

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit Team** - For the amazing web framework
- **scikit-learn** - Machine learning utilities
- **Plotly** - Interactive visualizations
- **ReportLab** - PDF generation capabilities

## 📞 Support

- 📧 Email: support@neurolog.dev
- 💬 Discord: [Join our community](https://discord.gg/neurolog)
- 🐛 Issues: [Report bugs](https://github.com/yourusername/neurolog/issues)
- � Docs: [Documentation](https://docs.neurolog.dev)

## 🌟 Show Your Support

⭐ If you find this project useful, please give it a star on GitHub!

---

**NeuroLog** - *Transform your logs into insights* 🚀
