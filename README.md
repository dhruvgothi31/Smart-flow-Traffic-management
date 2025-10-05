# SmartFlow Traffic System

An AI-powered traffic signal management system that uses computer vision and machine learning to optimize traffic flow in real-time.

## Features

- Real-time traffic monitoring and analysis
- AI-powered traffic signal optimization
- Interactive dashboard for traffic visualization
- Traffic pattern analytics
- Customizable system settings

## Project Structure

```
traffic-signal-ai/
├── README.md                     # Project documentation and setup instructions
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── app.py                        # Main Streamlit application
├── pages/                        # Streamlit pages
│   ├── dashboard.py              # Traffic monitoring dashboard
│   ├── analytics.py              # Traffic analysis page
│   └── settings.py               # System settings page
├── models/                       # ML models
│   ├── yolov8_model.py          # YOLOv8 implementation
│   └── weights/                  # Pre-trained weights folder
├── data/                         # Data storage
│   ├── raw/                      # Raw image/video data
│   ├── processed/                # Processed datasets
│   └── annotations/              # YOLO format annotations
└── scripts/                      # Utility scripts
    ├── annotate_data.py          # Script for image annotation
    ├── preprocess_data.py        # Data preparation script
    ├── train_model.py            # Model training script
    └── simulate_traffic.py       # Simple traffic simulation
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SmartFlow-Traffic-System.git
   cd SmartFlow-Traffic-System
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Other dependencies listed in requirements.txt

## Usage

1. Launch the application using `streamlit run app.py`
2. Access the dashboard through your web browser
3. Use the navigation menu to switch between different features:
   - Dashboard: Real-time traffic monitoring
   - Analytics: Traffic pattern analysis
   - Settings: System configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.