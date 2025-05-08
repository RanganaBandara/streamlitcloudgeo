import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class DistributionGenerator:
    def _init_(self):
        st.set_page_config(page_title="Distribution Generator", layout="wide")
        
        self.dist_functions = {
            "Normal Distribution": self._generate_normal,
            "Uniform Distribution": self._generate_uniform,
            "Log Normal Distribution": self._generate_lognormal,
            "Binomial Distribution": self._generate_binomial
        }
        
        # Initialize session state variables
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'dist_type' not in st.session_state:
            st.session_state.dist_type = "Normal Distribution"
            
    def create_layout(self):
        st.title("Distribution Generator")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._create_controls()
            
        with col2:
            self._create_visualization()
    
    def _on_dist_type_change(self):
        # This function will be called when distribution type changes
        st.session_state.params = {}
            
    def _create_controls(self):
        # Distribution Type outside the form for immediate updates
        st.subheader("Distribution Settings")
        st.selectbox(
            "Distribution Type",
            options=list(self.dist_functions.keys()),
            key='dist_type',
            on_change=self._on_dist_type_change
        )
        
        with st.form("distribution_form"):
            # UCS Range
            col1, col2 = st.columns(2)
            with col1:
                min_ucs = st.number_input("Minimum UCS (MPa)", value=0.0)
            with col2:
                max_ucs = st.number_input("Maximum UCS (MPa)", value=100.0)
            
            # Distribution Parameters
            st.subheader("Distribution Parameters")
            params = {}
            
            if st.session_state.dist_type == "Normal Distribution":
                col1, col2 = st.columns(2)
                with col1:
                    params['mean'] = st.number_input("Mean (μ)", value=50.0)
                with col2:
                    params['std'] = st.number_input("Std Dev (σ)", value=10.0)
                    
            elif st.session_state.dist_type == "Uniform Distribution":
                col1, col2 = st.columns(2)
                with col1:
                    params['low'] = st.number_input("Lower (a)", value=0.0)
                with col2:
                    params['high'] = st.number_input("Upper (b)", value=100.0)
                    
            elif st.session_state.dist_type == "Log Normal Distribution":
                col1, col2 = st.columns(2)
                with col1:
                    params['mean'] = st.number_input("Mean (μ)", value=1.0)
                with col2:
                    params['std'] = st.number_input("Std Dev (σ)", value=0.5)
                    
            elif st.session_state.dist_type == "Binomial Distribution":
                col1, col2 = st.columns(2)
                with col1:
                    params['n'] = st.number_input("Trials (n)", value=100, min_value=1)
                with col2:
                    params['p'] = st.number_input("Prob (p)", value=0.5, min_value=0.0, max_value=1.0)
            
            # Sample Size and Properties
            st.subheader("Sample Properties")
            sample_size = st.number_input("Sample Size", value=1000, min_value=1)
            diameter = st.number_input("Pile Diameter (m)", value=1.0)
            height = st.number_input("Height of Rock Socket (m)", value=1.0, min_value=0.0)
            

            
            # Coordinates Range
            col1, col2 = st.columns(2)
            with col1:
                x_range = st.number_input("X Range (m)", value=100.0)
            with col2:
                y_range = st.number_input("Y Range (m)", value=100.0)
            
            # Failure Analysis
            threshold = st.number_input("Failure Threshold", value=50.0)
            
            # Generate Button
            submitted = st.form_submit_button("Generate Data")
            
            if submitted:
                self._generate_distribution(
                    st.session_state.dist_type, params, sample_size, min_ucs, max_ucs,
                    diameter, x_range, y_range, threshold,height
                )
    
    def _create_visualization(self):
        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Data View", "Analysis"])
        
        with tab1:
            if st.session_state.generated_data is not None:
                self._plot_scatter()
                
        with tab2:
            if st.session_state.generated_data is not None:
                self._show_data()
                
        with tab3:
            if st.session_state.analysis_results is not None:
                self._show_analysis()
    
    def _generate_normal(self, size: int, params: dict) -> np.ndarray:
        random_probs = np.random.random(size)
        return norm.ppf(random_probs, params['mean'], params['std'])
    
    def _generate_uniform(self, size: int, params: dict) -> np.ndarray:
        return np.random.uniform(params['low'], params['high'], size)
    
    def _generate_lognormal(self, size: int, params: dict) -> np.ndarray:
        random_probs = np.random.random(size)
        return np.exp(norm.ppf(random_probs, params['mean'], params['std']))
    
    def _generate_binomial(self, size: int, params: dict) -> np.ndarray:
        random_probs = np.random.random(size)
        return binom.ppf(random_probs, params['n'], params['p'])
    
    def _generate_distribution(self, dist_type, params, size, min_ucs, max_ucs,
                             diameter, x_range, y_range, threshold,height):
        try:
            # Generate values
            values = self.dist_functions[dist_type](size, params)
            values = np.clip(values, min_ucs, max_ucs)
            
            # Calculate shaft resistance and skin friction
            shaft_resistance = 0.141 * np.power(values / 0.1, 0.5)
            skin_friction = shaft_resistance * np.pi * ((diameter / 2) ** 2) * height

            End_Bearing_capacity=4.66* np.power(values,0.56)

            Total_Load=End_Bearing_capacity+skin_friction

            
            # Generate coordinates
            x_coords = np.random.uniform(0, x_range, size)
            y_coords = np.random.uniform(0, y_range, size)
            
            # Calculate failure percentage
            failures = np.sum(Total_Load < threshold)
            failure_percentage = (failures / len(skin_friction)) * 100
            
            # Create DataFrame
            st.session_state.generated_data = pd.DataFrame({
                'X (m)': x_coords,
                'Y (m)': y_coords,
                'UCS of Rock': values,
                'End_bearing_capacity':End_Bearing_capacity,
                'Skin Friction': skin_friction,
                'total_Load':Total_Load,
                'Below Threshold': Total_Load< threshold
            })
            
            # Store analysis results
            st.session_state.analysis_results = {
                'total_samples': len(skin_friction),
                'failures': failures,
                'failure_percentage': failure_percentage,
                'threshold': threshold
            }
            
            # Save to Excel
            self._save_to_excel(dist_type)
            
        except Exception as e:
            st.error(f"Error generating distribution: {str(e)}")
    
    def _plot_scatter(self):
        df = st.session_state.generated_data
        
        fig = px.scatter(
            df,
            x='X (m)',
            y='Y (m)',
            color='Below Threshold',
            color_discrete_map={True: 'red', False: '#00BFFF'},
            labels={'Below Threshold': 'Status'},
            title='Sample Distribution'
        )
        
        fig.update_layout(
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#2b2b2b',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_data(self):
        df = st.session_state.generated_data
        st.dataframe(
            df.head(1000),
            height=800,  # Set the height to 600 pixels
            hide_index=True  # Optional: hide the index column for cleaner look
        )
        
        if len(df) > 1000:
            st.info("Showing first 1000 rows")
    
    def _show_analysis(self):
        results = st.session_state.analysis_results
        
        st.subheader("Analysis Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", f"{results['total_samples']:,}")
        with col2:
            st.metric("Failures", f"{results['failures']:,}")
        with col3:
            st.metric("Failure Percentage", f"{results['failure_percentage']:.2f}%")
    
    def _save_to_excel(self, dist_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"distribution_data_{timestamp}{dist_type.lower().replace(' ', '')}.xlsx"
        
        df = st.session_state.generated_data
        results = st.session_state.analysis_results
        
        analysis_df = pd.DataFrame({
            'Metric': ['Total Samples', 'Failures', 'Failure Percentage'],
            'Value': [
                results['total_samples'],
                results['failures'],
                f"{results['failure_percentage']:.2f}%"
            ]
        })
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            analysis_df.to_excel(writer, sheet_name='Failure Analysis', index=False)
        
        st.success(f"✓ Generated {filename}")

def main():
    app = DistributionGenerator()
    app.create_layout()

if __name__ == "_main_":
    main()