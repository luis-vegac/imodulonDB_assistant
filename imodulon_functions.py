import sys
import io
import difflib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

init_notebook_mode(connected=True) 

@tool
def learn_about_imodulons():
    
    """Returns detailed information about iModulons, their role in gene regulation, and their applications."""
    print("   Fetching information about imodulons")
    with open('data_files/about_imodulons.txt', 'r') as file:
        content = file.read()
    return content


@tool
def find_closest_imodulon(imodulon_names: list):
    
    """Finds the closest imodulons. Use to confirm user input imodulons exist"""
    
    print(f"\n   Finding closest imodulons to {imodulon_names}\n")
    imodulon_list = pd.read_csv('data_files/iM_table.csv').iloc[:, 0].tolist()
    imodulon_list_lower = [imodulon.lower() for imodulon in imodulon_list]
    result = {}
    for imodulon_name in imodulon_names:
        imodulon_name_lower = imodulon_name.lower()
        closest_matches = difflib.get_close_matches(imodulon_name_lower, imodulon_list_lower, n=3, cutoff=0.0)
        closest_matches_original_casing = [imodulon_list[imodulon_list_lower.index(match)] for match in closest_matches]
        if closest_matches_original_casing:
            result[imodulon_name] = closest_matches_original_casing
        else:
            result[imodulon_name] = "No imodulon match"
    return result


@tool
def find_closest_gene(gene_names: list):
    
    """Finds the closest genes. Use to confirm user input genes exist"""
    
    print(f"\n   Finding closest genes to {gene_names}\n")
    gene_list = pd.read_csv('data_files/gene_info.csv').iloc[:, 0].tolist()
    gene_list_lower = [str(gene).lower() for gene in gene_list]
    result = {}
    for gene_name in gene_names:
        gene_name_lower = gene_name.lower()
        closest_matches = difflib.get_close_matches(gene_name_lower, gene_list_lower, n=3, cutoff=0.0)
        closest_matches_original_casing = [gene_list[gene_list_lower.index(match)] for match in closest_matches]
        if closest_matches_original_casing:
            result[gene_name] = closest_matches_original_casing
        else:
            result[gene_name] = "No gene match"
    return result


@tool
def find_closest_condition(user_inputs: list):
    
    '''
    Finds the closest condition based on the specified criteria.
   
    Relevant columns and example entries:
       - strain description: ['Escherichia coli K-12 MG1655']
       - strain: ['MG1655']
       - culture type: ['Batch']
       - evolved sample: ['No']
       - base media: ['M9', 'LB']
       - temperature (c): [37]
       - ph: [7.0]
       - carbon source (g/l): ['glucose(2)']
       - nitrogen source (g/l): ['NH4Cl(1)']
       - electron acceptor: ['O2']
       - trace element mixture: ['sauer trace element mixture']
       - supplement: ['DPD (0.2mM)']
       - antibiotic for selection: ['Kanamycin (50 ug/mL)']
       
    The input MUST use the criteria in the input.
    Example usage:
       find_closest_condition(["base media: M9, supplement: FeCl2 (0.1mM)", "strain: MG1655, base media: LB"])
    '''
    
    print(f"\n   Finding closest conditions to {user_inputs}\n")
    def normalize_row(row):
        return {key.lower(): str(row[key]).lower() for key in user_input_dict.keys()}
    df = pd.read_csv('data_files/sample_table.csv')
    results = {}
    for user_input in user_inputs:
        user_input_dict = {}
        for pair in user_input.split(","):
            key, value = pair.split(":")
            user_input_dict[key.strip().lower()] = value.strip().lower()
        normalized_rows = df.apply(normalize_row, axis=1)
        closest_matches = []
        for index, row in normalized_rows.items():
            match_score = sum([user_input_dict[key] == row[key] for key in user_input_dict.keys()])
            closest_matches.append((index, match_score))
        closest_matches = sorted(closest_matches, key=lambda x: x[1], reverse=True)
        closest_conditions = df.iloc[[match[0] for match in closest_matches[:5]]]['condition'].tolist()
        if closest_conditions:
            results[user_input] = closest_conditions
        else:
            results[user_input] = "No condition match"
    return results


@tool
def get_genes_of_imodulons(imodulon_names: list):
    
    """Returns the genes of given imodulons."""
    
    print(f"\n   Getting genes of {imodulon_names} iModulons\n")
    df = pd.read_csv('data_files/gene_presence_list.csv')
    imodulon_list = df['iModulon'].unique().tolist()
    result = {}
    for imodulon_name in imodulon_names:
        if imodulon_name in imodulon_list:
           genes = df[df['iModulon'] == imodulon_name]['Gene'].tolist()
           result[imodulon_name] = genes if genes else "No genes found for the specified iModulon"
        else:
           result[imodulon_name] = f" {imodulon_name} iModulon not found. Please use find_closest_imodulon tool to double-check."
    return result


@tool
def get_condition_info(condition_names: list):
    
    """Returns the information of given experimental conditions."""
    
    print(f"\n   Retrieving info of {condition_names} conditions\n")
    df = pd.read_csv('data_files/sample_table.csv')
    condition_list = df['condition'].unique().tolist()
    
    result = {}
    for condition_name in condition_names:
        if condition_name in condition_list:
            condition_info = df[df['condition'] == condition_name]
            result[condition_name] = condition_info.to_dict(orient='records')[0] if not condition_info.empty else "No information found for the specified condition"
        else:
            result[condition_name] = f"{condition_name} condition not found. Please use find_closest_condition tool to double-check."
    return result


@tool
def get_gene_info(gene_names: list):
    """
    Returns the information of up to 4 given genes at a time.
    If more than 4 genes are provided, only the first 4 will be processed.
    """
    import pandas as pd

    # Limit to 4 genes
    gene_names = gene_names[:4]
    
    print(f"\nRetrieving info of {gene_names} genes\n")
    df = pd.read_csv('data_files/gene_info.csv')
    gene_list = df['gene name'].unique().tolist()
    
    result = {}
    for gene_name in gene_names:
        if gene_name in gene_list:
            gene_info = df[df['gene name'] == gene_name]
            result[gene_name] = gene_info.to_dict(orient='records')[0] if not gene_info.empty else "No information found for the specified gene"
        else:
            result[gene_name] = f"{gene_name} gene not found. Please use find_closest_gene tool to double-check before prompting the user again."
    return result



@tool
def get_imodulon_info(imodulon_names: list):
    
    """Returns the information of given imodulons."""

    print(f"\n   Retrieving info of {imodulon_names} imodulons\n")
    df = pd.read_csv('data_files/iM_table.csv')
    result = {}
    for imodulon_name in imodulon_names:
        imodulon_info = df[df['iModulon'] == imodulon_name]
        if not imodulon_info.empty:
            result[imodulon_name] = imodulon_info.to_dict(orient='records')[0]
        else:
            result[imodulon_name] = f"No information found for '{imodulon_name}'. Please use find_closest_imodulon tool to double-check before prompting the user again."
    return result


@tool
def plot_gene_expression(gene_name: str):
    """
    This function generates a bar plot showing the expression levels of the specified gene across the experimental conditions.
    The conditions are grouped by their studies, and detailed metadata is displayed when hovering over each bar.
    """
    print(f"\n   Plotting {gene_name} expression\n")
    
    # Load the activity matrix
    A = pd.read_csv('data_files/log_tpm.csv', index_col=0)
    if gene_name not in A.index:
        return f" {gene_name} gene not found. Please use find_closest_gene tool to double-check before prompting the user again."
    # Load the sample metadata
    sample_table = pd.read_csv('data_files/sample_table.csv', index_col=0)
    
    # Extract the gene activity levels
    gene_activities = A.loc[gene_name]
    
    # Ensure the sample IDs are in the same order
    gene_activities = gene_activities[sample_table.index]
    
    # Merge gene activities with sample metadata
    data = sample_table.copy()
    data['Activity'] = gene_activities.values
    data['index'] = data.index
    
    # Create a composite key for condition and study
    data['condition_study'] = data['condition'] + ' (' + data['study'] + ')'
    
    # Group by condition_study, condition, and study while preserving the order
    grouped_data = data.groupby(['condition_study', 'condition', 'study'], sort=False).agg({
        'Activity': ['mean', 'std', lambda x: ', '.join(map(str, x))],
        'strain description': 'first',
        'strain': 'first',
        'culture type': 'first',
        'evolved sample': 'first',
        'base media': 'first',
        'temperature (c)': 'first',
        'pH': 'first',
        'carbon source (g/L)': 'first',
        'nitrogen source (g/L)': 'first',
        'electron acceptor': 'first',
        'trace element mixture': 'first',
        'supplement': 'first',
        'antibiotic for selection': 'first',
        'growth rate (1/hr)': 'first',
        'isolate type': 'first',
        'additional details': 'first',
        'biological replicates': 'first',
        'doi': 'first',
        'run_date': 'first',
        'n_replicates': 'first'
    }).reset_index()
    
    # Flatten the MultiIndex columns
    grouped_data.columns = ['condition_study', 'condition', 'study', 'Activity_mean', 'Activity_std', 'Activities', 
                            'strain description', 'strain', 'culture type', 'evolved sample', 'base media', 'temperature (c)', 
                            'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor', 'trace element mixture', 
                            'supplement', 'antibiotic for selection', 'growth rate (1/hr)', 'isolate type', 'additional details', 
                            'biological replicates', 'doi', 'run_date', 'n_replicates']
    
    # Apply a minimum bar height for visibility and round numeric values
    min_visible_height = 0.1  # Minimum height for visibility
    grouped_data['Activity_mean'] = grouped_data['Activity_mean'].apply(lambda x: round(x if abs(x) > min_visible_height else min_visible_height * (1 if x > 0 else -1), 2))
    grouped_data['Activity_std'] = pd.to_numeric(grouped_data['Activity_std'], errors='coerce').apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    
    # Replace NaN values with empty strings
    grouped_data = grouped_data.fillna('')
    
    # Create a hover template
    hovertemplate = (
        "<b>Condition:</b> %{customdata[0]}<br>" +
        "<b>Mean Activity:</b> %{y:.2f}<br>" +
        "<b>Std Dev:</b> %{customdata[1]:.2f}<br>" +
        "<b>Activities:</b> %{customdata[2]}<br>" +
        "<b>Study:</b> %{customdata[3]}<br>" +
        "<b>Strain Description:</b> %{customdata[4]}<br>" +
        "<b>Strain:</b> %{customdata[5]}<br>" +
        "<b>Culture Type:</b> %{customdata[6]}<br>" +
        "<b>Evolved Sample:</b> %{customdata[7]}<br>" +
        "<b>Base Media:</b> %{customdata[8]}<br>" +
        "<b>Temperature (C):</b> %{customdata[9]}<br>" +
        "<b>pH:</b> %{customdata[10]}<br>" +
        "<b>Carbon Source (g/L):</b> %{customdata[11]}<br>" +
        "<b>Nitrogen Source (g/L):</b> %{customdata[12]}<br>" +
        "<b>Electron Acceptor:</b> %{customdata[13]}<br>" +
        "<b>Trace Element Mixture:</b> %{customdata[14]}<br>" +
        "<b>Supplement:</b> %{customdata[15]}<br>" +
        "<b>Antibiotic for selection:</b> %{customdata[16]}<br>" +
        "<b>Growth Rate (1/hr):</b> %{customdata[17]}<br>" +
        "<b>Isolate Type:</b> %{customdata[18]}<br>" +
        "<b>Additional Details:</b> %{customdata[19]}<br>" +
        "<b>Biological Replicates:</b> %{customdata[20]}<br>" +
        "<b>doi:</b> %{customdata[21]}<br>" +
        "<b>run_date:</b> %{customdata[22]}<br>" +
        "<b>n_replicates:</b> %{customdata[23]}<br>" +
        "<extra></extra>"
    )
    
    # Create the plot
    fig = go.Figure()
    color_discrete_map = px.colors.qualitative.Plotly
    projects = grouped_data['study'].unique()
    color_map = {project: color_discrete_map[i % len(color_discrete_map)] for i, project in enumerate(projects)}
    
    for project in projects:
        project_data = grouped_data[grouped_data['study'] == project]
        fig.add_trace(go.Bar(
            x=project_data['condition_study'],
            y=project_data['Activity_mean'],
            name=project,
            hovertemplate=hovertemplate,
            marker_color=color_map[project],  # Use distinct colors for each project
            customdata=project_data[['condition', 'Activity_std', 'Activities', 'study', 'strain description', 'strain', 'culture type', 
                                     'evolved sample', 'base media', 'temperature (c)', 'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor', 
                                     'trace element mixture', 'supplement', 'antibiotic for selection', 'growth rate (1/hr)', 'isolate type', 
                                     'additional details', 'biological replicates', 'doi', 'run_date', 'n_replicates']]
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='<b>Condition</b>',
        yaxis_title=f'<b>{gene_name} expression</b>',
        title={
            'text': f'<b>Gene Expression: {gene_name}</b>',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family='Arial, sans-serif', color='black')
        },
        barmode='stack',
        xaxis_tickangle=-90,
        width=1200,  # Make the plot wider
        height=600,  # Ensure the y-axis is fully displayed
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
        paper_bgcolor='rgba(245, 245, 245, 1)',  # Set paper background to a light grey
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        showlegend=False  # Remove the color legend
    )
    
    # Add vertical lines for project separation based on the order they appear in the grouped data
    project_boundaries = [0] + list(grouped_data.groupby((grouped_data['study'] != grouped_data['study'].shift()).cumsum()).size().cumsum())
    for boundary in project_boundaries[:-1]:  # Avoid adding a vertical line at the very end
        fig.add_vline(x=boundary - 0.5, line_width=1, line_dash="solid", line_color="lightgray")  # Continuous light gray lines
    
    # Update x-axis labels to only show project names at boundaries
    tickvals = [(project_boundaries[i] + project_boundaries[i+1]) / 2 for i in range(len(project_boundaries) - 1)]
    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=projects,
        fixedrange=False  # Fix the y-axis range so it doesn't change on zoom
    )
    
    fig.update_yaxes(fixedrange=True)
    
    # Update hover label appearance
    fig.update_traces(hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        font_color="black"
    ))
    
    iplot(fig, filename="gene_plot")
    return "Plot generated successfully"

@tool
def plot_imodulon_activity(imodulon_name: str):
    """
    This function creates a bar plot depicting the activity levels of the specified iModulon across experimental conditions.
    The conditions are grouped by their studies, and detailed metadata is shown when hovering over each bar.
    """
    print(f"\n   Plotting {imodulon_name} activity\n")
    
    # Load the activity matrix
    A = pd.read_csv('data_files/A.csv', index_col=0)
    if imodulon_name not in A.index:
        return f"{imodulon_name} iModulon not found. Please use find_closest_imodulon tool to double-check before prompting the user again."
    # Load the sample metadata
    sample_table = pd.read_csv('data_files/sample_table.csv', index_col=0)
    
    # Extract the gene activity levels
    imodulon_activities = A.loc[imodulon_name]
    
    # Ensure the sample IDs are in the same order
    imodulon_activities = imodulon_activities[sample_table.index]
    
    # Merge gene activities with sample metadata
    data = sample_table.copy()
    data['Activity'] = imodulon_activities.values
    data['index'] = data.index
    
    # Group by condition and study while preserving the order
    data['condition_study'] = data['condition'] + ' (' + data['study'] + ')'
    grouped_data = data.groupby(['condition_study', 'condition', 'study'], sort=False).agg({
        'Activity': ['mean', 'std', lambda x: ', '.join(map(str, x))],
        'strain description': 'first',
        'strain': 'first',
        'culture type': 'first',
        'evolved sample': 'first',
        'base media': 'first',
        'temperature (c)': 'first',
        'pH': 'first',
        'carbon source (g/L)': 'first',
        'nitrogen source (g/L)': 'first',
        'electron acceptor': 'first',
        'trace element mixture': 'first',
        'supplement': 'first',
        'antibiotic for selection': 'first',
        'growth rate (1/hr)': 'first',
        'isolate type': 'first',
        'additional details': 'first',
        'biological replicates': 'first',
        'doi': 'first',
        'run_date': 'first',
        'n_replicates': 'first'
    }).reset_index()
    
    # Flatten the MultiIndex columns
    grouped_data.columns = ['condition_study', 'condition', 'study', 'Activity_mean', 'Activity_std', 'Activities', 
                            'strain description', 'strain', 'culture type', 'evolved sample', 'base media', 'temperature (c)', 
                            'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor', 'trace element mixture', 
                            'supplement', 'antibiotic for selection', 'growth rate (1/hr)', 'isolate type', 'additional details', 
                            'biological replicates', 'doi', 'run_date', 'n_replicates']
    
    # Apply a minimum bar height for visibility and round numeric values
    min_visible_height = 0.1  # Minimum height for visibility
    grouped_data['Activity_mean'] = grouped_data['Activity_mean'].apply(lambda x: round(x if abs(x) > min_visible_height else min_visible_height * (1 if x > 0 else -1), 2))
    grouped_data['Activity_std'] = pd.to_numeric(grouped_data['Activity_std'], errors='coerce').apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    
    # Replace NaN values with empty strings
    grouped_data = grouped_data.fillna('')
    
    # Create a hover template
    hovertemplate = (
        "<b>Condition:</b> %{customdata[0]}<br>" +
        "<b>Mean Activity:</b> %{y:.2f}<br>" +
        "<b>Std Dev:</b> %{customdata[1]:.2f}<br>" +
        "<b>Activities:</b> %{customdata[2]}<br>" +
        "<b>Study:</b> %{customdata[3]}<br>" +
        "<b>Strain Description:</b> %{customdata[4]}<br>" +
        "<b>Strain:</b> %{customdata[5]}<br>" +
        "<b>Culture Type:</b> %{customdata[6]}<br>" +
        "<b>Evolved Sample:</b> %{customdata[7]}<br>" +
        "<b>Base Media:</b> %{customdata[8]}<br>" +
        "<b>Temperature (C):</b> %{customdata[9]}<br>" +
        "<b>pH:</b> %{customdata[10]}<br>" +
        "<b>Carbon Source (g/L):</b> %{customdata[11]}<br>" +
        "<b>Nitrogen Source (g/L):</b> %{customdata[12]}<br>" +
        "<b>Electron Acceptor:</b> %{customdata[13]}<br>" +
        "<b>Trace Element Mixture:</b> %{customdata[14]}<br>" +
        "<b>Supplement:</b> %{customdata[15]}<br>" +
        "<b>Antibiotic for selection:</b> %{customdata[16]}<br>" +
        "<b>Growth Rate (1/hr):</b> %{customdata[17]}<br>" +
        "<b>Isolate Type:</b> %{customdata[18]}<br>" +
        "<b>Additional Details:</b> %{customdata[19]}<br>" +
        "<b>Biological Replicates:</b> %{customdata[20]}<br>" +
        "<b>doi:</b> %{customdata[21]}<br>" +
        "<b>run_date:</b> %{customdata[22]}<br>" +
        "<b>n_replicates:</b> %{customdata[23]}<br>" +
        "<extra></extra>"
    )
    
    # Create the plot
    fig = go.Figure()
    color_discrete_map = px.colors.qualitative.Plotly
    projects = data['study'].unique()
    color_map = {project: color_discrete_map[i % len(color_discrete_map)] for i, project in enumerate(projects)}
    
    for project in projects:
        project_data = grouped_data[grouped_data['study'] == project]
        fig.add_trace(go.Bar(
            x=project_data['condition_study'],
            y=project_data['Activity_mean'],
            name=project,
            hovertemplate=hovertemplate,
            marker_color=color_map[project],  # Use distinct colors for each project
            customdata=project_data[['condition', 'Activity_std', 'Activities', 'study', 'strain description', 'strain', 'culture type', 
                                     'evolved sample', 'base media', 'temperature (c)', 'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor', 
                                     'trace element mixture', 'supplement', 'antibiotic for selection', 'growth rate (1/hr)', 'isolate type', 
                                     'additional details', 'biological replicates', 'doi', 'run_date', 'n_replicates']]
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='<b>Condition</b>',
        yaxis_title=f'<b>{imodulon_name} activity</b>',
        title={
            'text': f'<b>iModulon Activity: {imodulon_name}</b>',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family='Arial, sans-serif', color='black')
        },
        barmode='stack',
        xaxis_tickangle=-90,
        width=1200,  # Make the plot wider
        height=600,  # Ensure the y-axis is fully displayed
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
        paper_bgcolor='rgba(245, 245, 245, 1)',  # Set paper background to a light grey
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        showlegend=False  # Remove the color legend
    )
    
    # Add vertical lines for project separation based on the order they appear in the sample table
    project_boundaries = [0] + list(grouped_data.groupby((grouped_data['study'] != grouped_data['study'].shift()).cumsum()).size().cumsum())
    for boundary in project_boundaries[:-1]:  # Avoid adding a vertical line at the very end
        fig.add_vline(x=boundary - 0.5, line_width=1, line_dash="solid", line_color="lightgray")  # Continuous light gray lines
    
    # Update x-axis labels to only show project names at boundaries
    tickvals = [(project_boundaries[i] + project_boundaries[i+1]) / 2 for i in range(len(project_boundaries) - 1)]
    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=projects,
        fixedrange=False  # Fix the y-axis range so it doesn't change on zoom
    )
    
    fig.update_yaxes(fixedrange=True)
    
    # Update hover label appearance
    fig.update_traces(hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        font_color="black"
    ))
    
    iplot(fig, filename="imodulon_plot")
    return "Plot generated successfully\n"


@tool
def plot_all_imodulon_activities_for_condition(condition_name: str):
    """
    This function creates a bar plot depicting the activity levels of all iModulons in a specified condition.
    The activities are grouped by iModulons, and detailed metadata is shown when hovering over each bar.
    """
    print(f"\n   Plotting all iModulon activities for condition: {condition_name}\n")
    
    # Load the activity matrix
    A = pd.read_csv('data_files/A.csv', index_col=0)
    
    # Load the sample metadata
    sample_table = pd.read_csv('data_files/sample_table.csv', index_col=0)
    # Ensure the condition exists
    if condition_name not in sample_table['condition'].values:
        return f"Condition {condition_name} not found. Please use find_closest_condition tool to double-check before prompting the user again."
    
    # Get the sample IDs for the specified condition
    sample_ids = sample_table[sample_table['condition'] == condition_name].index
    
    # Extract the iModulon activity levels for the specified condition
    imodulon_activities = A[sample_ids].mean(axis=1)
    
    # Merge iModulon activities with iModulon names
    data = pd.DataFrame({
        'iModulon': A.index,
        'Activity': imodulon_activities.values
    })
    data = data.sort_values(by='iModulon')

    # Apply a minimum bar height for visibility
    min_visible_height = 0.1  # Minimum height for visibility
    data['Activity'] = data['Activity'].apply(lambda x: x if abs(x) > min_visible_height else min_visible_height * (1 if x > 0 else -1))
    
    # Create a hover template
    hovertemplate = (
        "<b>iModulon:</b> %{x}<br>" +
        "<b>Activity:</b> %{y:.2f}<br>" +
        "<extra></extra>"
    )
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['iModulon'],
        y=data['Activity'],
        hovertemplate=hovertemplate,
        marker_color='blue'
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='<b>iModulon</b>',
        yaxis_title=f'<b>{condition_name} activity</b>',
        title={
            'text': f'<b>iModulon Activities for Condition: {condition_name}</b>',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family='Arial, sans-serif', color='black')
        },
        barmode='stack',
        width=1200,  # Make the plot wider
        height=600,  # Ensure the y-axis is fully displayed
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
        paper_bgcolor='rgba(245, 245, 245, 1)',  # Set paper background to a light grey
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        showlegend=False  # Remove the color legend
    )
    
    fig.update_yaxes(fixedrange=True)
    
    # Update hover label appearance
    fig.update_traces(hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        font_color="black"
    ))
    
    iplot(fig, filename="all_imodulon_activities_for_condition_plot")
    return "Plot generated successfully\n"

@tool
def compare_gene_expression(gene1: str, gene2: str):
    """
    This function creates a scatter plot comparing the expression levels of two specified genes across various conditions.
    It includes a Pearson correlation line and displays detailed metadata when hovering over each point.
    """
    
    print(f"\n   Plotting {gene1} and {gene2} gene expression\n")
    # Load the activity matrix
    A = pd.read_csv('data_files/log_tpm.csv', index_col=0)
    if gene1 not in A.index:
        return f" {gene1} gene not found. Please use find_closest_gene tool to double-check before prompting the user again."
    if gene2 not in A.index:
        return f" {gene2} gene not found. Please use find_closest_gene tool to double-check before prompting the user again."

    # Load the sample metadata
    sample_table = pd.read_csv('data_files/sample_table.csv', index_col=0)
    
    # Extract the gene activity levels for both genes
    gene1_activities = A.loc[gene1]
    gene2_activities = A.loc[gene2]
    
    # Ensure the sample IDs are in the same order
    gene1_activities = gene1_activities[sample_table.index]
    gene2_activities = gene2_activities[sample_table.index]
    
    # Merge gene activities with sample metadata
    data = sample_table.copy()
    data[gene1] = gene1_activities.values
    data[gene2] = gene2_activities.values

    
    # Create composite key for condition and study
    data['condition_study'] = data['condition'] + ' (' + data['study'] + ')'
    
    # Group by condition and study and calculate mean, std, and individual values
    grouped_data = data.groupby(['condition_study', 'condition', 'study'], sort=False).agg({
        gene1: ['mean', 'std', lambda x: ', '.join(map(str, x))],
        gene2: ['mean', 'std', lambda x: ', '.join(map(str, x))],
        'strain description': 'first',
        'strain': 'first',
        'culture type': 'first',
        'evolved sample': 'first',
        'base media': 'first',
        'temperature (c)': 'first',
        'pH': 'first',
        'carbon source (g/L)': 'first',
        'nitrogen source (g/L)': 'first',
        'electron acceptor': 'first',
        'trace element mixture': 'first',
        'supplement': 'first',
        'antibiotic for selection': 'first',
        'growth rate (1/hr)': 'first',
        'isolate type': 'first',
        'additional details': 'first',
        'biological replicates': 'first',
        'doi': 'first',
        'run_date': 'first',
        'n_replicates': 'first'
    }).reset_index()
    
    # Flatten the MultiIndex columns
    grouped_data.columns = ['condition_study', 'condition', 'study', f'{gene1}_mean', f'{gene1}_std', f'{gene1}_values',
                            f'{gene2}_mean', f'{gene2}_std', f'{gene2}_values', 'strain description', 
                            'strain', 'culture type', 'evolved sample', 'base media', 'temperature (c)', 
                            'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor', 
                            'trace element mixture', 'supplement', 'antibiotic for selection', 'growth rate (1/hr)', 
                            'isolate type', 'additional details', 'biological replicates', 'doi', 'run_date', 'n_replicates']
    
    
    # Apply a minimum value for visibility and round numeric values
    min_visible_value = 0.1
    grouped_data[f'{gene1}_mean'] = grouped_data[f'{gene1}_mean'].apply(lambda x: round(x if abs(x) > min_visible_value else min_visible_value * (1 if x > 0 else -1), 2))
    grouped_data[f'{gene2}_mean'] = grouped_data[f'{gene2}_mean'].apply(lambda x: round(x if abs(x) > min_visible_value else min_visible_value * (1 if x > 0 else -1), 2))
    grouped_data[f'{gene1}_std'] = pd.to_numeric(grouped_data[f'{gene1}_std'], errors='coerce').apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    grouped_data[f'{gene2}_std'] = pd.to_numeric(grouped_data[f'{gene2}_std'], errors='coerce').apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    
    # Replace NaN values with empty strings
    grouped_data = grouped_data.fillna('')
    
    
    # Calculate Pearson correlation and linear regression on the averaged data
    r, p_value = pearsonr(grouped_data[f'{gene1}_mean'], grouped_data[f'{gene2}_mean'])
    slope, intercept, _, _, _ = linregress(grouped_data[f'{gene1}_mean'], grouped_data[f'{gene2}_mean'])
    
    
    # Create a scatter plot
    fig = px.scatter(
        grouped_data, x=f'{gene1}_mean', y=f'{gene2}_mean', color='study',
        labels={f'{gene1}_mean': f'{gene1} expression', f'{gene2}_mean': f'{gene2} expression'},
        title=f'Comparison of {gene1} and {gene2} Expression Levels'
    )
    
    # Add Pearson correlation line to the plot
    x_vals = np.array([grouped_data[f'{gene1}_mean'].min(), grouped_data[f'{gene1}_mean'].max()])
    y_vals = intercept + slope * x_vals
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name=f'Pearson R = {r:.2f}<br>p-value = {p_value:.2e}',
        line=dict(dash='dash', color='black')
    ))
    
   # Customize hover template
    fig.update_traces(hovertemplate=(
        "<b>Condition:</b> %{customdata[0]}<br>" +
        f"<b>{gene1} expression (mean):</b> %{{x:.2f}}<br>" +
        f"<b>{gene1} expression (std):</b> %{{customdata[1]:.2f}}<br>" +
        f"<b>{gene1} expressions:</b> %{{customdata[2]}}<br>" +
        f"<b>{gene2} expression (mean):</b> %{{y:.2f}}<br>" +
        f"<b>{gene2} expression (std):</b> %{{customdata[3]:.2f}}<br>" +
        f"<b>{gene2} expressions:</b> %{{customdata[4]}}<br>" +
        "<b>Study:</b> %{customdata[5]}<br>" +
        "<b>Strain Description:</b> %{customdata[6]}<br>" +
        "<b>Strain:</b> %{customdata[7]}<br>" +
        "<b>Culture Type:</b> %{customdata[8]}<br>" +
        "<b>Evolved Sample:</b> %{customdata[9]}<br>" +
        "<b>Base Media:</b> %{customdata[10]}<br>" +
        "<b>Temperature (C):</b> %{customdata[11]}<br>" +
        "<b>pH:</b> %{customdata[12]}<br>" +
        "<b>Carbon Source (g/L):</b> %{customdata[13]}<br>" +
        "<b>Nitrogen Source (g/L):</b> %{customdata[14]}<br>" +
        "<b>Electron Acceptor:</b> %{customdata[15]}<br>" +
        "<b>Trace Element Mixture:</b> %{customdata[16]}<br>" +
        "<b>Supplement:</b> %{customdata[17]}<br>" +
        "<extra></extra>"
      ), customdata=grouped_data[['condition', f'{gene1}_std', f'{gene1}_values', f'{gene2}_std', f'{gene2}_values', 'study',
                                'strain description', 'strain', 'culture type', 'evolved sample', 'base media',
                                'temperature (c)', 'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor',
                                'trace element mixture', 'supplement']]
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Comparison of {gene1} and {gene2} Expression Levels</b>',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family='Arial, sans-serif', color='black')
        },
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
        paper_bgcolor='rgba(245, 245, 245, 1)',  # Set paper background to a light grey
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        legend_title_text='Study',
        xaxis=dict(showgrid=True, gridcolor='lightgray', zerolinecolor='lightgray', zerolinewidth=1),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zerolinecolor='lightgray', zerolinewidth=1)
    )
    
    # Customize hover label appearance
    fig.update_traces(hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        font_color="black"
    ))
    
    iplot(fig, filename="genes plot")
    return "Plot generated successfully\n"



    
@tool
def compare_imodulon_activities(imodulon1: str, imodulon2: str):
    """
    This function generates a scatter plot comparing the activity levels of two specified iModulons across experimental conditions.
    It includes a Pearson correlation line and shows detailed metadata when hovering over each point.
    """
    print(f"\n   Plotting {imodulon1} and {imodulon2} imodulon activities\n")
    
    # Load the activity matrix
    A = pd.read_csv('data_files/A.csv', index_col=0)
    if imodulon1 not in A.index:
        return f" {imodulon1} gene not found. Please use find_closest_gene tool to double-check before prompting the user again."
    if imodulon2 not in A.index:
        return f" {imodulon2} gene not found. Please use find_closest_gene tool to double-check before prompting the user again."

    # Load the sample metadata
    sample_table = pd.read_csv('data_files/sample_table.csv', index_col=0)
    
    # Extract the imodulon activity levels for both imodulons
    imodulon1_activities = A.loc[imodulon1]
    imodulon2_activities = A.loc[imodulon2]
    
    # Ensure the sample IDs are in the same order
    imodulon1_activities = imodulon1_activities[sample_table.index]
    imodulon2_activities = imodulon2_activities[sample_table.index]
    
    # Merge imodulon activities with sample metadata
    data = sample_table.copy()
    data[imodulon1] = imodulon1_activities.values
    data[imodulon2] = imodulon2_activities.values

    # Create composite key for condition and study
    data['condition_study'] = data['condition'] + ' (' + data['study'] + ')'
    
    # Group by condition and study and calculate mean, std, and individual values
    grouped_data = data.groupby(['condition_study', 'condition', 'study'], sort=False).agg({
        imodulon1: ['mean', 'std', lambda x: ', '.join(map(str, x))],
        imodulon2: ['mean', 'std', lambda x: ', '.join(map(str, x))],
        'strain description': 'first',
        'strain': 'first',
        'culture type': 'first',
        'evolved sample': 'first',
        'base media': 'first',
        'temperature (c)': 'first',
        'pH': 'first',
        'carbon source (g/L)': 'first',
        'nitrogen source (g/L)': 'first',
        'electron acceptor': 'first',
        'trace element mixture': 'first',
        'supplement': 'first',
        'antibiotic for selection': 'first',
        'growth rate (1/hr)': 'first',
        'isolate type': 'first',
        'additional details': 'first',
        'biological replicates': 'first',
        'doi': 'first',
        'run_date': 'first',
        'n_replicates': 'first'
    }).reset_index()
    
    # Flatten the MultiIndex columns
    grouped_data.columns = ['condition_study', 'condition', 'study', f'{imodulon1}_mean', f'{imodulon1}_std', f'{imodulon1}_values',
                            f'{imodulon2}_mean', f'{imodulon2}_std', f'{imodulon2}_values', 'strain description', 
                            'strain', 'culture type', 'evolved sample', 'base media', 'temperature (c)', 
                            'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor', 
                            'trace element mixture', 'supplement', 'antibiotic for selection', 'growth rate (1/hr)', 
                            'isolate type', 'additional details', 'biological replicates', 'doi', 'run_date', 'n_replicates']
    
    # Apply a minimum value for visibility and round numeric values
    min_visible_value = 0.1
    grouped_data[f'{imodulon1}_mean'] = grouped_data[f'{imodulon1}_mean'].apply(lambda x: round(x if abs(x) > min_visible_value else min_visible_value * (1 if x > 0 else -1), 2))
    grouped_data[f'{imodulon2}_mean'] = grouped_data[f'{imodulon2}_mean'].apply(lambda x: round(x if abs(x) > min_visible_value else min_visible_value * (1 if x > 0 else -1), 2))
    grouped_data[f'{imodulon1}_std'] = pd.to_numeric(grouped_data[f'{imodulon1}_std'], errors='coerce').apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    grouped_data[f'{imodulon2}_std'] = pd.to_numeric(grouped_data[f'{imodulon2}_std'], errors='coerce').apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    
    # Replace NaN values with empty strings
    grouped_data = grouped_data.fillna('')
    
    # Calculate Pearson correlation and linear regression on the averaged data
    r, p_value = pearsonr(grouped_data[f'{imodulon1}_mean'], grouped_data[f'{imodulon2}_mean'])
    slope, intercept, _, _, _ = linregress(grouped_data[f'{imodulon1}_mean'], grouped_data[f'{imodulon2}_mean'])
    
    # Create a scatter plot
    fig = px.scatter(
        grouped_data, x=f'{imodulon1}_mean', y=f'{imodulon2}_mean', color='study',
        labels={f'{imodulon1}_mean': f'{imodulon1} activity', f'{imodulon2}_mean': f'{imodulon2} activity'},
        title=f'Comparison of {imodulon1} and {imodulon2} Activity Levels'
    )
    
    # Add Pearson correlation line to the plot
    x_vals = np.array([grouped_data[f'{imodulon1}_mean'].min(), grouped_data[f'{imodulon1}_mean'].max()])
    y_vals = intercept + slope * x_vals
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name=f'Pearson R = {r:.2f}<br>p-value = {p_value:.2e}',
        line=dict(dash='dash', color='black')
    ))
    
    # Customize hover template
    fig.update_traces(hovertemplate=(
        "<b>Condition:</b> %{customdata[0]}<br>" +
        f"<b>{imodulon1} activity (mean):</b> %{{x:.2f}}<br>" +
        f"<b>{imodulon1} activity (std):</b> %{{customdata[1]:.2f}}<br>" +
        f"<b>{imodulon1} activities:</b> %{{customdata[2]}}<br>" +
        f"<b>{imodulon2} activity (mean):</b> %{{y:.2f}}<br>" +
        f"<b>{imodulon2} activity (std):</b> %{{customdata[3]:.2f}}<br>" +
        f"<b>{imodulon2} activities:</b> %{{customdata[4]}}<br>" +
        "<b>Study:</b> %{customdata[5]}<br>" +
        "<b>Strain Description:</b> %{customdata[6]}<br>" +
        "<b>Strain:</b> %{customdata[7]}<br>" +
        "<b>Culture Type:</b> %{customdata[8]}<br>" +
        "<b>Evolved Sample:</b> %{customdata[9]}<br>" +
        "<b>Base Media:</b> %{customdata[10]}<br>" +
        "<b>Temperature (C):</b> %{customdata[11]}<br>" +
        "<b>pH:</b> %{customdata[12]}<br>" +
        "<b>Carbon Source (g/L):</b> %{customdata[13]}<br>" +
        "<b>Nitrogen Source (g/L):</b> %{customdata[14]}<br>" +
        "<b>Electron Acceptor:</b> %{customdata[15]}<br>" +
        "<b>Trace Element Mixture:</b> %{customdata[16]}<br>" +
        "<b>Supplement:</b> %{customdata[17]}<br>" +
        "<extra></extra>"
    ), customdata=grouped_data[['condition', f'{imodulon1}_std', f'{imodulon1}_values', f'{imodulon2}_std', f'{imodulon2}_values', 'study',
                                'strain description', 'strain', 'culture type', 'evolved sample', 'base media',
                                'temperature (c)', 'pH', 'carbon source (g/L)', 'nitrogen source (g/L)', 'electron acceptor',
                                'trace element mixture', 'supplement']]
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Comparison of {imodulon1} and {imodulon2} Activity Levels</b>',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family='Arial, sans-serif', color='black')
        },
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
        paper_bgcolor='rgba(245, 245, 245, 1)',  # Set paper background to a light grey
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        legend_title_text='Study',
        xaxis=dict(showgrid=True, gridcolor='lightgray', zerolinecolor='lightgray', zerolinewidth=1),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zerolinecolor='lightgray', zerolinewidth=1)
    )
    
    # Customize hover label appearance
    fig.update_traces(hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        font_color="black"
    ))
    
    iplot(fig, filename="imodulons plot")
    return "Plot generated succesfully\n"
    
@tool
def plot_dima(condition1: str, condition2: str):
    """
    This function creates a scatter plot comparing the activity levels of all iModulons between two specified conditions.
    Outliers are highlighted and labeled, and a Pearson correlation line is included.
    Hovering over each point displays the iModulon name and activity levels.
    """
    print(f"\n   Plotting Differential iModulon Activity plot of {condition1} and {condition2} imodulons\n")
    
    # Load the activity matrix
    A = pd.read_csv('data_files/A.csv', index_col=0)
    # Load the sample metadata
    sample_table = pd.read_csv('data_files/sample_table.csv', index_col=0)
    # Load the iModulon metadata
    iM_table = pd.read_csv('data_files/iM_table.csv', index_col=0)
    
    # Filter sample_table to only include the specified conditions
    sample_table_filtered = sample_table[(sample_table['condition'] == condition1) | (sample_table['condition'] == condition2)]
    sample_table_filtered = sample_table_filtered.reset_index()
    # Group the sample_table to calculate the mean and std for each condition
    grouped_sample_table = sample_table_filtered.groupby(['condition', 'study']).agg({
        'sample_id': 'count'
    }).reset_index()
    
    # Extract the iModulon activity levels for the specified conditions and average them
    condition1_activities = A[sample_table[sample_table['condition'] == condition1].index].mean(axis=1)
    condition2_activities = A[sample_table[sample_table['condition'] == condition2].index].mean(axis=1)
    
    # Merge the data with the iModulon table
    data = pd.DataFrame({
        'iModulon': A.index,
        condition1: condition1_activities.values,
        condition2: condition2_activities.values
    }).merge(iM_table[['function']], left_on='iModulon', right_index=True)
    
    # Calculate Pearson correlation and linear regression
    r, p_value = pearsonr(data[condition1], data[condition2])
    slope, intercept, _, _, _ = linregress(data[condition1], data[condition2])
    
    # Identify outliers
    threshold = 5  # Set a threshold for outliers
    outliers = data[(np.abs(data[condition1] - data[condition2]) > threshold)]
    
    # Create a scatter plot
    fig = go.Figure()
    
    # Add non-outliers
    non_outliers = data[~data.index.isin(outliers.index)]
    fig.add_trace(go.Scatter(
        x=non_outliers[condition1],
        y=non_outliers[condition2],
        mode='markers',
        name='Non-Outliers',
        marker=dict(color='gray', size=5),
        hovertemplate=(
            "<b>iModulon:</b> %{text}<br>" +
            f"<b>{condition1} activity:</b> %{{x:.2f}}<br>" +
            f"<b>{condition2} activity:</b> %{{y:.2f}}<br>"
            "<extra></extra>"
        ),
        text=non_outliers['iModulon']
    ))
    
    # Add outliers
    fig.add_trace(go.Scatter(
        x=outliers[condition1],
        y=outliers[condition2],
        mode='markers+text',
        text=outliers['iModulon'],
        textposition='top center',
        name='Outliers',
        marker=dict(size=10, color='blue'),
        customdata=np.stack((outliers['iModulon'], outliers['function']), axis=-1),
        hovertemplate=(
            "<b>iModulon:</b> %{customdata[0]}<br>" +
            "<b>Function:</b> %{customdata[1]}<br>" +
            f"<b>{condition1} activity:</b> %{{x:.2f}}<br>" +
            f"<b>{condition2} activity:</b> %{{y:.2f}}<br>"
            "<extra></extra>"
        )
    ))
    
    # Add Pearson correlation line to the plot
    x_vals = np.array([data[condition1].min(), data[condition1].max()])
    y_vals = intercept + slope * x_vals
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name=f'Pearson R = {r:.2f}<br>p-value = {p_value:.2e}',
        line=dict(dash='dash', color='black')
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Comparison of iModulon Activity: {condition1} vs {condition2}</b>',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family='Arial, sans-serif', color='black')
        },
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
        paper_bgcolor='rgba(245, 245, 245, 1)',  # Set paper background to a light grey
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        xaxis=dict(title=f'{condition1} activity', showgrid=True, gridcolor='lightgray', zerolinecolor='lightgray', zerolinewidth=1),
        yaxis=dict(title=f'{condition2} activity', showgrid=True, gridcolor='lightgray', zerolinecolor='lightgray', zerolinewidth=1)
    )
    
    # Customize hover label appearance
    fig.update_traces(hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        font_color="black"
    ))
    
    # Show the legend for Pearson R and p-value
    fig.update_layout(
        showlegend=True,
        legend_title_text='Legend'
    )
    
    iplot(fig, filename="dima plot")
    return "Plot generated successfully\n"

@tool
def execute_python_code(code: str):
    
    """
    Executes the given Python code. The data has already been loaded into the following variables:
        imodulon_list_df, gene_list_df, sample_table_df, gene_presence_df, iM_table_df, log_tpm_df, A_df

    Args:
    code (str): The Python code to be executed.

    Returns:
    str: The output or result of the executed code, or an error message if execution fails.
    
    Example usage:
    execute_python_code("print('Hello, World!')")
    """
    
    print(f"\n   Executing Python code:\n{code}\n")
    imodulon_list_df = pd.read_csv('data_files/iM_table.csv')
    gene_list_df = pd.read_csv('data_files/gene_info.csv')
    sample_table_df = pd.read_csv('data_files/sample_table.csv')
    gene_presence_df = pd.read_csv('data_files/gene_presence_list.csv')
    iM_table_df = pd.read_csv('data_files/iM_table.csv')
    log_tpm_df = pd.read_csv('data_files/log_tpm.csv', index_col=0)
    A_df = pd.read_csv('data_files/A.csv', index_col=0)
    
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        exec(code)
        result = new_stdout.getvalue()
    except Exception as e:
        result = str(e)
    finally:
        sys.stdout = old_stdout
    
    return result
