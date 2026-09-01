## Overview ##
Updated PLQY analysis code to extract PLQY, OD, and FWHM values from data collected in C31. Data files and metadata (integration times) are inputted via a gui.

## Inputs ##
The input file must be in the format .txt. <BR>
The file must contain WAVELENGTH and INTENSITY columns. Any other columns are ignored. <BR>
The <CODE>utils</CODE> module contains a function <CODE>find_data_start_row</CODE> that will determine the start row of the numerical data, allowing headers of variable length to be parsed. <BR>

## Pre-requisites ##
You will need <CODE>python >= 3.11</CODE> and <CODE>uv</CODE> which can be installed at: https://docs.astral.sh/uv/getting-started/installation/

## Usage ##
To run the analysis code:
<LI>First clone the repository by typing the following into your terminal: <CODE>git clone https://github.com/sophietukcer/absorption_analysis.git</CODE></LI>
<LI>To install the dependencies in a virtual environment, <CODE>venv</CODE>, run: <CODE>uv sync</CODE></LI>
<LI>Finally, to run the analysis, run: <CODE>uv run main.py</CODE>. A gui will pop up to input data files.</LI>

## Future work ##
<LI>Add functionality to analyses multiple files from one folder.</LI>
<LI>Add feature to enable alternative file naming for <CODE>_out</CODE> measurements, for instances where multiple spot <CODE>_in</CODE> measurements use the same <CODE>_out</CODE> measurement.</LI>
