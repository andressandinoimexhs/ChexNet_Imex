from fpdf import FPDF
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import pandas as pd

from GenerateReportConstantManager import thoraxlabels
from GenerateReportConstantManager import OutputPath_Heatmap, OutputPath_pdfreport
from GenerateReportConstantManager import Path_TemplateUS, Path_TemplateLATAM, Path_OutputReportGenerated


def generate_predictedlabels(prediction):
    
    """
    Sorted predicted labels from highest to lowest prediction values
    """
    
    # Create dataframe two colums (Labels,Predictions)
    ThoraxDataFrame=pd.DataFrame(columns=['Labels','Predictions'])
    
    # Colum1: Labels
    ThoraxDataFrame['Labels'] = thoraxlabels
    
    # Colum2: Predictions
    ThoraxDataFrame['Predictions'] = prediction
    
    # Sort from highest to lowest prediction values
    ThoraxDataFrame.sort_values(by=['Predictions'],
                                ascending=False)
    
    # Subsetting the first three values
    ThoraxDataFrameSubset = ThoraxDataFrame.iloc[:3]
    Sort_ThoraxDataFrame = ThoraxDataFrameSubset.sort_values(by=['Predictions'],
                                                             ascending=False)
    
    
    # Predictions and labels organized in a short list
    
    LabelList = []
    PredictionList = []
    
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
    
    # Predictions and labels organized in a short list
    
    label1 = predictionlist[0]
    pred1  = predictionlist[1]
    label2 = predictionlist[2]
    pred2  = predictionlist[3]
    label3 = predictionlist[4]
    pred3  = predictionlist[5]
    
    border=0
    # 1. Set up the PDF doc basics
    pdf = FPDF('P', 'cm', 'Letter')

    # Adding new page
    pdf.add_page()
    
    # Set up document margins
    pdf.set_left_margin(1.5)

    pdf.ln(2)
    
    pdf.set_font('Arial', 'BU',28)
    pdf.set_text_color(32, 32,91)
    #pdf.set_fill_color(173, 197, 231)
    pdf.set_fill_color(173, 197, 250)
    pdf.cell(18, 1.5, 'Chest X-RAY Report', 0, 0,'C', fill=1)
    pdf.set_font('Arial', '', 12)
    
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(18, 0.5,'Patient information',align='C',border=0)
    
    pdf.ln(0.5)
    pdf.line(pdf.x, pdf.y, pdf.x+18, pdf.y)
    
    pdf.ln(0.7)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(5, 0.5,'ID',align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(5, 0.5,'Name',align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(4, 0.5,'Birth date',align='C',border=border)
    
    x_position = pdf.x
    pdf.x = x_position
    pdf.cell(4, 0.5,'Gender',align='C',border=border)
    
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
    
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(32, 32,91)
    pdf.ln(1)
    pdf.cell(18, 0.5,'Study Information',align='C',border=border)
    pdf.ln(0.5)
    pdf.line(pdf.x, pdf.y, pdf.x+18, pdf.y)
    
    #pdf.ln(0.5)
    pdf.set_font('Arial', '', 12)
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
    
    pdf.image(OutputPath_Heatmap,x=12,y=11,w=7.5,h=7.5)
    
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
    
    
    """
    Salida del reporte en pdf (Sin fondo)
    """
    pdf.output(OutputPath_pdfreport, 'F')
    
    return 

def get_pdfreport(region):
    
    PdfBaseReport = PdfFileReader(OutputPath_pdfreport,'rb')
    
    if region=='US':
    
        PdfBackground = PdfFileReader(Path_TemplateUS,'rb')
    
    else:
        
        PdfBackground = PdfFileReader(Path_TemplateLATAM,'rb')
    
    PdfBaseReport = PdfBaseReport.getPage(0)
    PdfBackground = PdfBackground.getPage(0)

    PdfBaseReport.mergePage(PdfBackground)
    pdfOutput = PdfFileWriter()
    pdfOutput.addPage(PdfBaseReport)
    
    """
     Salida del reporte en pdf (Con fondo) [Path_OutputReportGenerated]
    """
    pdfOutputFile = open(Path_OutputReportGenerated, 'wb')
    
    pdfOutput.write(pdfOutputFile)
    
    pdfOutputFile.close()
    
    return

    

