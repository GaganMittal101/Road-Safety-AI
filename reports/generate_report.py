import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_pdf_report(state, selected_data, predicted_acc, predicted_risk):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"Road_Safety_Report_{state}_{datetime.now().strftime('%Y%m%d')}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Road Safety AI - Report")
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 80, f"State: {state}")
    c.drawString(50, height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 140, "Accident Data")
    c.setFont("Helvetica", 12)
    y = height - 170
    for year in ["2019", "2020", "2021", "2022", "2023"]:
        val = selected_data[f"{year} Accidents"].values[0]
        c.drawString(50, y, f"{year}: {int(val)}")
        y -= 20

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y - 20, "Prediction")
    c.setFont("Helvetica", 12)
    y -= 50
    c.drawString(50, y, f"Predicted Accidents (2024): {predicted_acc}")
    y -= 20
    c.drawString(50, y, f"Risk Level: {predicted_risk}")

    y -= 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Recommendations")
    c.setFont("Helvetica", 12)
    y -= 30
    recs = [
        "Improve road infrastructure",
        "Increase traffic monitoring",
        "Install AI cameras",
        "Conduct awareness campaigns"
    ]
    for r in recs:
        c.drawString(50, y, f"• {r}")
        y -= 20

    c.save()
    return pdf_path