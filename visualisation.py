import streamlit as st
import pandas as pd
import numpy as np
import duckdb

# 1. Establish connection to your database file
conn = duckdb.connect('db/SGJobData_Normalized.db')

# 2. Define your SQL query based on your business objective
jobs_count_sql = """select count(*) as job_count 
                    from Jobs
                """

categories_sql = """select sector,Cat_Name as Category
                    from Categories 
                    order by sector, Cat_Name;
                """

sector_jobs_sql = """select c.Sector, count(*) as 'Number Of Job Postings'
                    from JobCategories jc
                    inner join Categories c
                    on jc.Cat_ID = c.Cat_ID
                    group by c.Sector
                    order by count(*) desc;
                """

sector_jobs_status_sql = """
                        select c.Sector, j.status_jobStatus as 'Job Status', count(*) as 'Number Of Job Postings'
                        from JobCategories jc
                        inner join Categories c on jc.Cat_ID = c.Cat_ID
                        inner join Jobs j on jc.metadata_jobPostId = j.metadata_jobPostId
                        group by c.Sector, j.status_jobStatus
                        order by c.Sector;
                    """

exp_bins_sql = """
                SELECT 
                    CASE 
                        WHEN minimumYearsExperience BETWEEN 0 AND 5 THEN '0-5'
                        WHEN minimumYearsExperience BETWEEN 6 AND 10 THEN '6-10'
                        WHEN minimumYearsExperience BETWEEN 11 AND 15 THEN '11-15'
                        WHEN minimumYearsExperience BETWEEN 16 AND 20 THEN '16-20'
                        WHEN minimumYearsExperience BETWEEN 21 AND 25 THEN '21-25'
                        WHEN minimumYearsExperience BETWEEN 26 AND 30 THEN '26-30'
                        WHEN minimumYearsExperience BETWEEN 31 AND 35 THEN '31-35'
                        WHEN minimumYearsExperience BETWEEN 36 AND 40 THEN '36-40'
                        ELSE '40+'
                    END AS ExpBin,
                    minimumYearsExperience,
                    COUNT(*) AS CountInBin
                FROM Jobs
                GROUP BY 
                    CASE 
                        WHEN minimumYearsExperience BETWEEN 0 AND 5 THEN '0-5'
                        WHEN minimumYearsExperience BETWEEN 6 AND 10 THEN '6-10'
                        WHEN minimumYearsExperience BETWEEN 11 AND 15 THEN '11-15'
                        WHEN minimumYearsExperience BETWEEN 16 AND 20 THEN '16-20'
                        WHEN minimumYearsExperience BETWEEN 21 AND 25 THEN '21-25'
                        WHEN minimumYearsExperience BETWEEN 26 AND 30 THEN '26-30'
                        WHEN minimumYearsExperience BETWEEN 31 AND 35 THEN '31-35'
                        WHEN minimumYearsExperience BETWEEN 36 AND 40 THEN '36-40'
                        ELSE '40+'
                    END,
                    minimumYearsExperience
                ORDER BY minimumYearsExperience;
            """

# 3. Load the result set directly into a Pandas DataFrame
jobs_count_df = conn.query(jobs_count_sql).to_df()
categories_df = conn.query(categories_sql).to_df()
sectors_df = categories_df.groupby('Sector')['Category'].apply(lambda x: ', '.join(x)).reset_index()

sector_jobs_df = conn.query(sector_jobs_sql).to_df()
sector_jobs_df['% of Total Job Postings'] = sector_jobs_df['Number Of Job Postings'] / sector_jobs_df['Number Of Job Postings'].sum()
sector_jobs_df = sector_jobs_df[['Sector','% of Total Job Postings','Number Of Job Postings']]

sector_jobs_status_df = conn.query(sector_jobs_status_sql).to_df()
sector_jobs_status_df = sector_jobs_status_df.pivot(index='Sector', columns='Job Status', values='Number Of Job Postings').reset_index()
print(sector_jobs_status_df)

sector_jobs_status_combined = pd.merge(sector_jobs_df, sector_jobs_status_df, on='Sector', how='inner')
sector_jobs_status_combined['% of Open Postings'] = sector_jobs_status_combined['Open'] / sector_jobs_status_combined['Number Of Job Postings']
sector_jobs_status_combined['% of Closed Postings'] = sector_jobs_status_combined['Closed'] / sector_jobs_status_combined['Number Of Job Postings']
sector_jobs_status_combined['% of Re-open Postings'] = sector_jobs_status_combined['Re-open'] / sector_jobs_status_combined['Number Of Job Postings']
sector_jobs_status_combined = sector_jobs_status_combined[['Sector','% of Total Job Postings','Number Of Job Postings','Open','% of Open Postings','Closed','% of Closed Postings','Re-open','% of Re-open Postings']]
print(sector_jobs_status_combined)

exp_bins_df = conn.query(exp_bins_sql).to_df()
exp_bins_df = exp_bins_df.groupby('ExpBin', sort=False)['CountInBin'].sum().reset_index()
exp_bins_df['Sequence'] = np.arange(9)

showVisualisation = True
if showVisualisation:
    # 4. Streamlit Visualisation
    # Sets the page configuration
    # You can set the page title and layout here
    st.set_page_config(page_title="Singapore Jobs Analytics: Government Insights", layout="wide")
    st.title("Singapore Jobs Analytics: Government Insights")
    st.caption("NTU SCTP DSAI GROUP 1")

    st.header("Key Metrics")
    # Create four columns for the metrics and unpack them
    # We can then use each column to place a metric
    col1, col2, col3, col4 = st.columns(4)

    # Populate each column with a metric by passing label and value
    col1.metric("Total Job Postings", f"{jobs_count_df['job_count'].item():,}")
    col2.metric("Number Of Sectors (Categories)", f"{len(sectors_df)} ({len(categories_df)})")
    col3.metric("Total Job Postings By Sectors", f"{sector_jobs_df['Number Of Job Postings'].sum():,}")

    st.subheader("Job Categories By Sectors")
    st.dataframe(sectors_df, hide_index=True, height="content")

    st.subheader("Job Postings By Sectors and Statuses")
    st.dataframe(sector_jobs_status_combined.style.format(
                                                {'Number Of Job Postings': '{:,}',
                                                 'Open': '{:,}',
                                                 'Closed': '{:,}',
                                                 'Re-open': '{:,}',
                                                 '% of Total Job Postings' : '{:.2%}',
                                                 '% of Open Postings' : '{:.2%}',
                                                 '% of Closed Postings' : '{:.2%}',
                                                 '% of Re-open Postings' : '{:.2%}'
                                                 }
                                            )
                , hide_index=True, width="content", height="content")

    #col_left, col_right = st.columns(2)
    # Tells Streamlit to put the following content in the left column
    #with col_left:           
        #st.subheader("XXX)

    # Tells Streamlit to put the following content in the right column
    #with col_right:
        #st.subheader("XXX")

    
    st.subheader("All Job By Years Of Experience")
    st.dataframe(exp_bins_df, hide_index=True, height="content")
    st.bar_chart(exp_bins_df, x="ExpBin", y="CountInBin", sort="Sequence",
                horizontal=False, width=2000, height="content")

    # postings by month/year over time, stacked bar by experience levels

# Close connection
conn.close()