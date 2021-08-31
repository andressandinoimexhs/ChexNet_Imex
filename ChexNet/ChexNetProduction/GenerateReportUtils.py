# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:56:51 2021

@author: Andres
"""
from fpdf import FPDF
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import pandas as pd

def generate_predictedlabels(prediction):
    
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
    
    ThoraxDataFrame=pd.DataFrame(columns=['Labels','Predictions'])
    ThoraxDataFrame['Labels'] = thoraxlabels
    ThoraxDataFrame['Predictions'] = prediction
    ThoraxDataFrame.sort_values(by=['Predictions'],
                                ascending=False)
    
    ThoraxDataFrameSubset = ThoraxDataFrame.iloc[:3]
    Sort_ThoraxDataFrame = ThoraxDataFrameSubset.sort_values(by=['Predictions'],
                                                             ascending=False)
    
    LabelList=[]
    PredictionList=[]

    for i in range(3):
    
        Label = Sort_ThoraxDataFrame.iloc[i][0]
        LabelList.append(Label)
        Prediction = str(round(Sort_ThoraxDataFrame.iloc[i][1]*100,1))+" %"
        PredictionList.append(Prediction)
            
    PredictionListOutput = [LabelList[0],PredictionList[0],
                            LabelList[1],PredictionList[1],
                            LabelList[2],PredictionList[2]]

    return PredictionListOutput


def generate_pdftemplate(patient_name,
                         ID,
                         genre,
                         date,
                         study_name,
                         study_date,
                         report,
                         predictionlist):
    
    label1 = predictionlist[0]
    pred1  = predictionlist[1]
    label2 = predictionlist[2]
    pred2  = predictionlist[3]
    label3 = predictionlist[4]
    pred3  = predictionlist[5]
    
    border=1
    # 1. Set up the PDF doc basics
    pdf = FPDF('P', 'cm', 'Letter')

    pdf.add_page()
    pdf.set_left_margin(1.5)

    pdf.ln(2)
    
    pdf.set_font('Arial', 'BU',28)
    pdf.set_text_color(32, 32,91)
    pdf.set_fill_color(173, 197, 231)
    
    pdf.cell(18, 1.5, 'Chest X-RAY Report', 0, 0,'C', fill=1)
    pdf.set_font('Arial', '', 12)
    
    pdf.ln(2)
    pdf.cell(18, 0.5,'Patient information',align='C',border=0)
    
    pdf.ln(0.5)
    pdf.line(pdf.x, pdf.y, pdf.x+18, pdf.y)
    
    pdf.ln(0.7)
    
    pdf.cell(5, 0.5,'ID',align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(5, 0.5,'Name',align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(4, 0.5,'Birth date',align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(4, 0.5,'Genre',align='C',border=border)
    
    pdf.set_text_color(0,0,0)
    pdf.ln(1)
    pdf.cell(5, 0.5,ID,align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(5, 0.5,patient_name,align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(4, 0.5,date,align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(4, 0.5,genre,align='C',border=border)
    
    pdf.set_text_color(32, 32,91)
    pdf.ln(1)
    pdf.cell(18, 0.5,'Study Information',align='C',border=border)
    pdf.ln(0.5)
    pdf.line(pdf.x, pdf.y, pdf.x+18, pdf.y)
    
    #pdf.ln(0.5)
    pdf.cell(18, 0.5,'',align='C',border=border)
    pdf.ln(0.5)
    
    pdf.cell(5, 0.5,'Name',align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(5, 0.5,'Date',align='C',border=border)
    
    pdf.set_text_color(0, 0,0)
    pdf.ln(1)
    pdf.cell(5, 0.5,study_name,align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(5, 0.5,study_date,align='C',border=border)
        
    pdf.ln(1)
    pdf.set_text_color(32, 32,91)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(10, 1,'Evaluation description',align='L',border=border)
    pdf.set_text_color(0,0,0)
    
    pdf.ln(1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(10,0.7,report,align='L',border=border)
    
    pdf.image('./misc/thoraxheatmap.png',x=12,y=11,w=7.5,h=7.5)
    
    y_position = 19
    pdf.y = y_position
    
    pdf.set_text_color(32, 32,91)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(10, 0.5,'Computer Assisted Diagnosis (CAD)',align='L',border=border)
    
    pdf.ln(1)
    
    pdf.set_text_color(0,0,0) #DarkBlue
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(10,0.7, txt=label1 + ": "+ pred1 ,align='L',border=border)
    pdf.multi_cell(10,0.7, txt=label2 + ": "+ pred2 ,align='L',border=border)
    pdf.multi_cell(10,0.7, txt=label3 + ": "+ pred3 ,align='L',border=border)
    
    pdf.output('./misc/fpdf_pdf_report.pdf', 'F')
    
    return 

def get_pdfreport(region):
    
    PdfBaseReport = PdfFileReader('./misc/fpdf_pdf_report.pdf','rb')
    
    if region=='US':
    
        PdfBackground = PdfFileReader('./misc/IMEXHSUS.pdf','rb')
    
    else:
        
        PdfBackground = PdfFileReader('./misc/IMEXHSLATAM.pdf','rb')
    
    PdfBaseReport=PdfBaseReport.getPage(0)
    PdfBackground=PdfBackground.getPage(0)

    PdfBaseReport.mergePage(PdfBackground)
    pdfOutput = PdfFileWriter()
    pdfOutput.addPage(PdfBaseReport)
    
    pdfOutputFile = open('./misc/OutputReport.pdf', 'wb')
    
    pdfOutput.write(pdfOutputFile)
    
    pdfOutputFile.close()
    
    return

    

