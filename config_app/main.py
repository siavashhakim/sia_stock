import panel as pn
from lambda_stock_ui import SessionManager
from app_manager import ConfigAppManager
from stock_forecasting import StockForecasting
# from financial_analysis import FinancialAnalysis
import os

os.environ['working_bucket'] = 'crcdal-well-data'
base_s3_bucket = 's3://stock-forecasting-sia/'
StockForecasting.s3_bucket_path = base_s3_bucket
# FinancialAnalysis.s3_bucket_path = base_s3_bucket
app = SessionManager().start_session(ConfigAppManager)

# Serve the app
app.servable()
pn.serve(app)
