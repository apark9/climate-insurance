'''
TERMINAL COMMANDS:

bsub -Is -q short_int -n 8 -R "rusage[mem=16G]" python main.py
tail -f main.log

pip freeze > requirements.txt
'''

interview_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Insurance_Interviews/"
transcript_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/data/Public_Insurance_Transcripts/"
output_folder = "/export/home/rcsguest/rcs_apark/Desktop/home-insurance/output/"

keyword_flag = 'financial'

if keyword_flag == 'financial':
    keywords = [
        'catastrophe', 'underwriting losses', 'rate increases', 'premium increases',
        'reinsurance', 'claims reserves', 'risk modeling', 'loss ratios',
        'carbon pricing', 'sustainability initiatives', 'ESG investing'
    ]
elif keyword_flag == 'climate':
    keywords = [
        'climate change', 'natural disasters', 'extreme weather', 'hurricane',
        'wildfire', 'earthquake', 'flood', 'coastal flooding', 'sea level rise',
        'drought', 'adaptation strategies', 'climate resilience', 'GHG emissions',
        'green energy transition', 'net zero commitments'
    ]

# Key Line Items Related to Climate Risk, Mitigation, and Geographic Exposure for AFG

key_line_items_afg = [
    # Climate Risk Indicators
    "Property and transportation loss and LAE catastrophe losses, mm",
    "Specialty casualty loss and LAE catastrophe losses, mm",
    "Specialty financial loss and LAE catastrophe losses, mm",
    "Property and transportation loss and LAE excl. catastrophe and prior year development, mm",
    "Specialty casualty loss and LAE excl. catastrophe and prior year development, mm",
    "Specialty financial loss and LAE excl. catastrophe and prior year development, mm",
    "Total Loss and LAE, mm",
    "Property and transportation loss and LAE catastrophe losses ratio, %",
    "Specialty casualty loss and LAE catastrophe losses ratio, %",
    "Specialty financial loss and LAE catastrophe losses ratio, %",
    "Total Loss and LAE Ratio, %",

    # Mitigation Efforts
    "Reinsurance and other receivables, mm",
    "Payable to reinsurers, mm",
    "Ceded annuity receipts",
    "Ceded annuity surrenders, benefits, and withdrawals",
    "Property and transportation combined ratio, %",
    "Specialty casualty combined ratio, %",
    "Specialty financial combined ratio, %",
    "Total Combined Ratio, %",
    "Adjusted Combined Ratio, %",
    "Property and transportation net retention ratio, %",
    "Specialty casualty net retention ratio, %",
    "Specialty financial net retention ratio, %",
    "Total Net Retention Ratio, %",
    "Total Other Operating Expense Ratio, %",

    # Geographic Exposure
    "Property and transportation gross written premiums, mm",
    "Specialty casualty gross written premiums, mm",
    "Specialty financial gross written premiums, mm",
    "Total Gross Written Premiums, mm",
    "Property and transportation net written premiums, mm",
    "Specialty casualty net written premiums, mm",
    "Specialty financial net written premiums, mm",
    "Total Net Written Premiums, mm",
    "Property and transportation net earned premiums, mm",
    "Specialty casualty net earned premiums, mm",
    "Specialty financial net earned premiums, mm",
    "Total Net Earned Premiums, mm",
    "Property and transportation loss and LAE, mm",
    "Specialty casualty loss and LAE, mm",
    "Specialty financial loss and LAE, mm",
    "Property and transportation other operating expense, mm",
    "Specialty casualty other operating expense, mm",
    "Specialty financial other operating expense, mm",

    # Additional Key Metrics
    "Net Unearned Premium Reserves",
    "Net Loss Reserves",
    "Reserve Ratio, %",
    "Solvency Ratio, %",
    "Total Net Revenue",
    "Consensus Estimates - Total Net Earned Premiums, mm",
    "Consensus Estimates - Total Underwriting Income, mm",
    "Core Net Operating Earnings",
    "Adjusted Combined Ratio, %"
]

# Key Line Items Related to Climate Risk, Mitigation, and Geographic Exposure for TRV

key_line_items_trv = [
    # Climate Risk Indicators
    "Business Insurance - Catastrophes, net of reinsurance, mm",
    "Bond & Specialty Insurance - Catastrophes, net of reinsurance, mm",
    "Personal Insurance - Catastrophes, net of reinsurance, mm",
    "Total Catastrophes, net of reinsurance, mm",
    "Business Insurance loss and LAE excl. PYD & CAT, mm",
    "Bond & Specialty Insurance loss and LAE excl. PYD & CAT, mm",
    "Personal Insurance loss and LAE excl. PYD & CAT, mm",
    "Total loss and LAE excl. PYD & CAT, mm",
    "Business Insurance loss and LAE ratio excl. PYD & CAT, %",
    "Bond & Specialty Insurance loss and LAE ratio excl. PYD & CAT, %",
    "Personal Insurance loss and LAE ratio excl. PYD & CAT, %",
    "Total loss and LAE ratio excl. PYD & CAT, %",
    "Catastrophes, net of reinsurance, %",
    "Total Loss and LAE Ratio, %",

    # Mitigation Efforts
    "Reinsurance recoverables",
    "Payables for reinsurance premiums",
    "Business Insurance combined ratio, %",
    "Bond & Specialty Insurance combined ratio, %",
    "Personal Insurance combined ratio, %",
    "Total Combined Ratio, %",
    "Total Combined Ratio excl. PYD & CAT, %",
    "Deferred Acquisition Costs and Value of Business Acquired",
    "Unearned Premium Reserves",
    "Net Loss Reserves",
    "Reserve Ratio, %",
    "Solvency Ratio, %",
    "Core Income",
    "Adjusted Combined Ratio, %",

    # Geographic Exposure
    "Business Insurance gross written premiums, mm",
    "Bond & Specialty Insurance gross written premiums, mm",
    "Personal Insurance gross written premiums, mm",
    "Total Gross Written Premiums, mm",
    "Business Insurance net written premiums, mm",
    "Bond & Specialty Insurance net written premiums, mm",
    "Personal Insurance net written premiums, mm",
    "Total Net Written Premiums, mm",
    "Business Insurance net earned premiums, mm",
    "Bond & Specialty Insurance net earned premiums, mm",
    "Personal Insurance net earned premiums, mm",
    "Total Net Earned Premiums, mm",
    "Net Earned Premiums",
    "Net Investment Income",
    "Net Investment Gains",

    # Additional Key Metrics
    "Consensus Estimates - Total Net Earned Premiums, mm",
    "Consensus Estimates - Total Underwriting Income excl. PYD & CAT, mm",
    "Consensus Estimates - Total Combined Ratio excl. PYD & CAT, %",
    "Consensus Estimates - Net Revenue",
    "Consensus Estimates - EBT",
    "Net CFO",
    "Net CFI",
    "Net CFF",
    "Return on Average Common Equity, %",
    "Consensus Estimates - Return on Average Common Equity, %",
    "Book Value per Common Share",
    "Consensus Estimates - Book Value per Common Share",
    "Effective Interest Rate on Debt, %"
]

# Key Line Items Related to Climate Risk, Mitigation, and Geographic Exposure for AFL

key_line_items_afl = [
    # Climate Risk Indicators
    "Aflac Japan - Benefits and claims, mm",
    "Aflac U.S - Benefits and claims, mm",
    "Corporate and Other - Benefits and claims, mm",
    "Total Policyholder Benefits, Claims and Dividends, mm",
    "Future Policy Benefits",
    "Deferred Acquisition Costs and Value of Business Acquired",

    # Mitigation Efforts
    "Deferred Acquisition Costs",
    "Total Deferred Acquisition Costs, mm",
    "Aflac Japan - Insurance commissions, mm",
    "Aflac U.S - Insurance commissions, mm",
    "Total Insurance Commissions, mm",
    "Reinsurance recoverables",
    "Effective Interest Rate on Debt, %",

    # Geographic Exposure
    "Aflac Japan - Premiums, mm",
    "Aflac U.S - Premiums, mm",
    "Corporate and Other - Premiums, mm",
    "Total Premiums, mm",
    "Net Earned Premiums",
    "Net Investment Income",
    "Net Investment Gains",

    # Additional Key Metrics
    "Net Revenue",
    "Consensus Estimates - Net Revenue",
    "Return on Average Common Equity, %",
    "Consensus Estimates - Return on Average Common Equity, %",
    "Book Value per Common Share",
    "Consensus Estimates - Book Value per Common Share",
    "Debt to Capital Ratio, %"
]

# Key Line Items Related to Climate Risk, Mitigation, and Geographic Exposure for PGR

key_line_items_pgr = [
    # Climate Risk Indicators
    "Total Loss and LAE, mm",
    "Y/Y Total Loss and LAE growth, %",
    "Unearned Premium Reserves",
    "Loss Reserves",
    "Net Loss Reserves",
    "Reserve Ratio, %",
    "Solvency Ratio, %",
    "Future Policy Benefits",
    "Deferred Acquisition Costs and Value of Business Acquired",

    # Mitigation Efforts
    "Deferred acquisition costs",
    "Reinsurance recoverable",
    "Prepaid reinsurance premiums",
    "Payables for reinsurance premiums",
    "Unearned premiums",
    "Loss and loss adjustment expense reserves",
    "Total Underwriting Expense, mm",
    "Underwriting Margin, %",
    "Y/Y Improvement in Underwriting Margin, bps",

    # Geographic Exposure
    "Net Earned Premiums",
    "Total Net Written Premiums, mm",
    "Total Net Earned Premiums, mm",
    "Net Earned Premium Mix, %",
    "Business Insurance gross written premiums, mm",
    "Total Gross Written Premiums, mm",
    "Net Investment Income",
    "Net Investment Gains",

    # Additional Key Metrics
    "Adjusted Earnings Per Share (No Adjustments) - WAD",
    "Consensus Estimates - Adjusted Earnings Per Share",
    "Book Value per Common Share",
    "Consensus Estimates - Book Value per Common Share",
    "Effective Interest Rate on Debt, %",
    "Consensus Estimates - Net Revenue",
    "Return on Average Common Equity, %",
    "Consensus Estimates - Return on Average Common Equity, %"
]

# Key Line Items Related to Climate Risk, Mitigation, and Geographic Exposure for ALL

key_line_items_all = [
    # Climate Risk Indicators
    "Allstate Protection - Auto - Effect of catastrophe losses, mm",
    "Allstate Protection - Homeowners - Effect of catastrophe losses, mm",
    "Allstate Protection - Other personal lines - Effect of catastrophe losses, mm",
    "Allstate Protection - Commercial lines - Effect of catastrophe losses, mm",
    "Total Property & Liability Loss and LAE, mm",
    "Gross Loss Reserves, mm",
    "Reinsurance Recoverable, mm",
    "Net Loss Reserves, mm",
    "Loss Payout Ratio, %",

    # Mitigation Efforts
    "Deferred policy acquisition costs",
    "Payables for reinsurance premiums",
    "Deferred Acquisition Costs",
    "Total Deferred Acquisition Costs, mm",
    "Allstate Protection combined ratio, %",
    "Consensus Estimates - Property & Liability - Combined Ratio, %",
    "Effective Interest Rate on Debt, %",

    # Geographic Exposure
    "Allstate Protection - Auto - Net premiums written, mm",
    "Allstate Protection - Homeowners - Net premiums written, mm",
    "Allstate Protection - Other personal lines - Net premiums written, mm",
    "Allstate Protection - Commercial lines - Net premiums written, mm",
    "Total Property & Liability Net Written Premiums, mm",
    "Allstate Protection - Auto - Net premiums earned, mm",
    "Allstate Protection - Homeowners - Net premiums earned, mm",
    "Allstate Protection - Other personal lines - Net premiums earned, mm",
    "Allstate Protection - Commercial lines - Net premiums earned, mm",
    "Total Property & Liability Net Earned Premiums, mm",
    "Net Earned Premiums",
    "Net Investment Income",
    "Net Investment Gains",

    # Additional Key Metrics
    "Consensus Estimates - Total Property & Liability Net Earned Premiums, mm",
    "Consensus Estimates - Total Property & Liability Underwriting Income, mm",
    "Return on Average Common Equity, %",
    "Consensus Estimates - Return on Average Common Equity, %",
    "Book Value per Common Share",
    "Consensus Estimates - Book Value per Common Share",
    "Debt to Capital Ratio, %"
]
