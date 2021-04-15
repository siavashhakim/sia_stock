from lambda_stock_ui import SessionManager
from lambda_stock_ui import AppManager
from stock_forecasting import StockForecasting
from financial_analysis import FinancialAnalysis
from stock_index import StockIndex


class ConfigAppManager(AppManager):
    init_instructions = {
        'Stock Forecasting': {
            'class': StockForecasting,
            'filename': ''
        },
        'Financial Analysis': {
            'class': FinancialAnalysis,
            'filename': ''
        },
        'Stock Index': {
            'class': StockIndex,
            'filename': ''
        }


    }

    def __init__(self, id):
        """
        Args:
            id:
        """
        super().__init__(id)
        self.field_obj = None
        self.base_s3_bucket = 's3://stcok-forecasting-sia/'



