# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:56:51 2021

@author: Andres
"""
from fpdf import FPDF
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import pandas as pd

def generate_predictedlabels(prediction):

    thoraxlabels = ["Atelectasis","Cardiomegaly",
                    "Effusion","Infiltration",
                    "Mass","Nodule","Pneumonia",
                    "Pneumothorax","Consolidation",
                    "Edema","Emphysema","Fibrosis",
                    "Pleural_Thickening","Hernia"]
    
    df=pd.DataFrame(columns=['Labels','Predictions'])
    df['Labels'] = thoraxlabels
    df['Predictions'] = prediction
    df.sort_values(by=['Predictions'],ascending=False)
    
    df2 = df.iloc[:3]
    df3=df2.sort_values(by=['Predictions'],ascending=False)
    
    label1 = df3.iloc[0][0]
    pred1 = str(round(df3.iloc[0][1]*100,1))+" %"
    
    label2 = df3.iloc[1][0]
    pred2 = str(round(df3.iloc[1][1]*100,1))+" %"
    
    label3 = df3.iloc[2][0]
    pred3 = str(round(df3.iloc[2][1]*100,1))+" %"
    
    predlist = [label1,pred1,label2,pred2,label3,pred3]
    
    return predlist

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
    
    border=0
    # 1. Set up the PDF doc basics
    pdf = FPDF('P', 'cm', 'Letter')
    #pdf.open(,'')
    #pdf.set_margins(0.5,1.5,2.5)
    pdf.add_page()
    pdf.set_left_margin(1.5)
    #pdf.image('C:/Users/Andres/Desktop/latam.jpeg',x=0,y=0,w=21.59,h=27.94)
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
    
    # y_position = 10
    # #pdf.y = y_position
    # x_position = 15
    #pdf.y = y_position
    
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
    
    
    
    pdf1Reader = PdfFileReader('./misc/fpdf_pdf_report.pdf','rb')
    
    if region=='US':
        #US
        pdf2Reader = PdfFileReader('./misc/IMEXHSUS.pdf','rb')
    
    else:
        pdf2Reader = PdfFileReader('./misc/IMEXHSLATAM.pdf','rb')
    
    aa=pdf1Reader.getPage(0)
    bb=pdf2Reader.getPage(0)
    
    #pdfWriter = PdfFileWriter()
    pdfWriter = PdfFileMerger()
    aa.mergePage(bb)
    output_pdf = PdfFileWriter()
    output_pdf.addPage(aa)
    
    pdfOutputFile = open('./misc/OutputReport.pdf', 'wb')
    output_pdf.write(pdfOutputFile)
    
    pdfOutputFile.close()
    return

    

