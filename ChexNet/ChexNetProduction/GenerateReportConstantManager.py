import numpy as np


"""
Constants for GenerateReportUtils.py
"""
# Constants for generate_predictedlabels() method

thoraxlabels = ["Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Pleural_Thickening",
                "Hernia"]

# Constants for generate_pdftemplate() method

OutputPath_Heatmap = './misc/thoraxheatmap.png'
OutputPath_pdfreport = './misc/fpdf_pdf_report.pdf'

Path_TemplateUS = './misc/IMEXHSUS.pdf'
Path_TemplateLATAM = './misc/IMEXHSUS.pdf'
Path_OutputReportGenerated = './misc/OutputReport.pdf'