from GenerateReportUtils import generate_pdftemplate,get_pdfreport
from GenerateReportUtils import generate_pdftemplate,generate_predictedlabels

class GenerateReportClass:
    
  def __init__(self, patient_name, ID, genre,date,study_name,study_date,report,predictions,region):
    self.patient_name = patient_name
    self.ID = ID
    self.genre = genre
    self.date = date
    self.study_name = study_name
    self.study_date = study_date
    self.report = report
    self.predictions = predictions
    self.region = region

  def generate_pdf(self):
      
    predictedlabels=generate_predictedlabels(self.predictions)
    
    generate_pdftemplate(self.patient_name,
                         self.ID,
                         self.genre,
                         self.date,
                         self.study_name,
                         self.study_date,
                         self.report,
                         predictedlabels
                         )
    
    get_pdfreport(self.region)
    
    return