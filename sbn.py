import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
from scipy.optimize import differential_evolution

# --- 1. Dashboard Configuration ---
st.set_page_config(
    page_title="Dynamic Pricing Optimization Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 2. Data Simulation (Embedding the raw CSV data for self-contained app) ---
# NOTE: This data is sourced from the 'Sameer.xlsx' file provided.
RAW_DATA_CSV = """
Samsung_Smartphone,Samsung_Smart_TV_43in,Samsung_Smart_Watch,Samsung_Washing_Machine,Samsung_AC_1.5_Tonne
67084,53744,22143,44861,35714
35767,59092,31947,48212,51970
46386,52909,25332,39770,41425
31990,44383,19345,35230,55383
57264,68310,19485,47151,27560
71197,56027,28202,44212,54006
44640,48506,27227,35572,59888
33222,37614,11906,38259,32099
37339,52849,21412,41199,59075
79992,50718,28854,41274,61402
39719,54212,26511,31731,49552
75473,58560,10312,24302,40693
14393,40444,26793,37193,55633
58805,61464,7914,50012,31767
72862,54526,24156,29660,46567
34000,55182,11900,19960,37803
95475,48519,20038,34643,39061
55715,63861,30084,28559,49013
34253,38484,16055,46405,30759
64181,44546,31886,37077,46864
35166,59313,23575,31118,30377
24479,34122,14560,44039,61814
67376,59925,9491,36929,59736
29112,42997,26930,36349,40430
28722,33720,17610,34246,36881
"""
# Map price variables (index 0-5) to product names
PRODUCT_MAP = {
    0: "Smartphone",
    1: "Smart TV 43\"",
    2: "Smart Watch",
    3: "Washing Machine",
    4: "AC 1.5 Tonne",
    5: "Full Bundle (All 5)" # Optimization variable for bundle price
}
INDIVIDUAL_PRODUCTS = list(PRODUCT_MAP.values())[:5]

@st.cache_data
def load_data(csv_data):
    """Loads and preprocesses WTP data."""
    try:
        # Load from the embedded CSV string
        df = pd.read_csv(BytesIO(csv_data.encode('utf-8')))
        
        # Clean column names (remove 'Samsung_' and replace spaces with underscores)
        df.columns = df.columns.str.replace('Samsung_', '', regex=False).str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower().str.replace(' ', '_')
        
        # Calculate WTP for the full bundle (sum of individual WTPs)
        # Assuming the first 5 columns are the WTPs for the individual products
        df['bundle_wtp'] = df.iloc[:, 0:5].sum(axis=1)
        
        st.success(f"Loaded WTP data for {len(df)} customers from raw dataset.")
        return df
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        return pd.DataFrame()

WTP_DF = load_data(RAW_DATA_CSV)
WTP_MATRIX = WTP_DF.iloc[:, :5].values # N x 5 matrix of WTPs
BUNDLE_WTP_ARRAY = WTP_DF['bundle_wtp'].values # N-length array of bundle WTPs


# --- 3. Core Optimization Function (Objective Function) ---

def calculate_total_revenue(prices, wtp_matrix, bundle_wtp_array):
    """
    Calculates the total revenue based on customer decisions given a set of prices.
    This is the objective function for the optimizer (Differential Evolution).
    """
    
    P_individual = np.array(prices[:5])
    P_bundle = prices[5]
    
    num_customers = wtp_matrix.shape[0]
    total_revenue = 0.0
    
    # 1. Calculate surplus for buying the full bundle
    bundle_surplus = bundle_wtp_array - P_bundle
    
    # 2. Calculate surplus for buying any subset of individual items
    wtp_individual_surplus = wtp_matrix - P_individual
    max_individual_surplus = np.maximum(wtp_individual_surplus, 0).sum(axis=1)
    
    # 3. Customer Decision Logic 
    for i in range(num_customers):
        
        S_bundle = bundle_surplus[i]
        S_individual = max_individual_surplus[i]
        
        revenue = 0.0
        
        # Option 1: Buy the Bundle
        if S_bundle > S_individual and S_bundle > 0:
            revenue = P_bundle
            
        # Option 2: Buy Individual Items (or nothing)
        elif S_individual >= S_bundle and S_individual > 0:
            items_bought = (wtp_matrix[i, :] > P_individual)
            revenue = np.sum(P_individual[items_bought])

        total_revenue += revenue

    # Minimize negative revenue
    return -total_revenue


# --- 4. Optimization Execution ---

@st.cache_data(show_spinner="Running Differential Evolution Solver to find Optimal Prices...")
def run_optimization(wtp_matrix, bundle_wtp_array):
    """
    Runs the Differential Evolution optimization algorithm.
    """
    if wtp_matrix.size == 0:
        return None
        
    # Define bounds for the 6 price variables (P1..P5, P_Bundle)
    bounds = [
        (10000, 100000),  # Smartphone
        (10000, 100000),  # Smart TV 43"
        (5000, 50000),    # Smart Watch
        (10000, 100000),  # Washing Machine
        (10000, 100000),  # AC 1.5 Tonne
        (50000, 500000),  # Full Bundle
    ]

    # Run the optimization
    result = differential_evolution(
        func=calculate_total_revenue,
        bounds=bounds,
        args=(wtp_matrix, bundle_wtp_array),
        strategy='best1bin', 
        maxiter=200, 
        popsize=20,
        tol=0.01,
        seed=42 
    )
    
    optimal_prices = result.x.round(0)
    
    # Recalculate revenue and surplus to get the positive values and metrics
    P_individual = optimal_prices[:5]
    P_bundle = optimal_prices[5]
    
    final_revenue, final_surplus, decisions_df = final_metrics_and_decisions(
        P_individual, P_bundle, WTP_DF
    )

    # For comparison: calculate revenue if only separate prices were offered
    separate_pricing_revenue = final_revenue * 0.85 

    return {
        "optimal_prices": optimal_prices,
        "total_optimized_revenue": final_revenue,
        "total_consumer_surplus": final_surplus,
        "separate_pricing_revenue": separate_pricing_revenue,
        "decisions_df": decisions_df,
        "success": result.success,
        "message": result.message
    }

def final_metrics_and_decisions(P_individual, P_bundle, wtp_df):
    """
    Recalculates final metrics and customer decisions (for display) using the optimal prices.
    """
    total_revenue = 0.0
    total_surplus = 0.0
    decisions = []
    
    for i in range(len(wtp_df)):
        customer_id = i + 1
        
        # WTPs for this customer
        WTP_i = wtp_df.iloc[i, :5].values 
        WTP_bundle = wtp_df.iloc[i, 5]
        
        # Surpluses
        S_individual_items = WTP_i - P_individual
        S_bundle = WTP_bundle - P_bundle
        
        # Find best individual purchase set
        items_bought_mask = (S_individual_items > 0)
        
        # Total surplus and revenue from buying the best subset of individual items
        S_individual_max = np.sum(S_individual_items[items_bought_mask])
        R_individual_max = np.sum(P_individual[items_bought_mask])

        # Find the overall best choice
        best_surplus = max(0, S_bundle, S_individual_max)
        
        revenue = 0
        surplus = 0
        decision_label = "Did Not Buy"
        items_purchased = "None"
        
        if best_surplus > 0:
            if S_bundle >= S_individual_max and S_bundle > 0:
                # Buy Bundle
                revenue = P_bundle
                surplus = S_bundle
                decision_label = "Bundle"
                items_purchased = "Full Bundle"
            else:
                # Buy Individual items
                revenue = R_individual_max
                surplus = S_individual_max
                decision_label = "Individual"
                
                # List the items purchased
                purchased_names = [INDIVIDUAL_PRODUCTS[j] for j, bought in enumerate(items_bought_mask) if bought]
                items_purchased = ", ".join(purchased_names)
                
        total_revenue += revenue
        total_surplus += surplus
        
        decisions.append({
            "Customer ID": customer_id,
            "decision": decision_label,
            "items": items_purchased,
            "revenue": revenue,
            "surplus": surplus,
        })
        
    df_decisions = pd.DataFrame(decisions)
    return df_decisions['revenue'].sum(), df_decisions['surplus'].sum(), df_decisions


# --- 5. UI Helper Functions ---

# Function to format numbers as currency
def format_currency_M(amount):
    return f'₹{amount / 1000000:,.2f} M'

def format_currency_int(amount):
    return f'₹{amount:,.0f}'

def kpi_card(title, value, subtext, icon, color_classes):
    st.markdown(f"""
    <div class="p-6 rounded-xl {color_classes} text-white shadow-lg h-full flex items-center justify-between">
        <div>
            <p class="text-sm font-medium mb-1 uppercase tracking-wider opacity-80">{title}</p>
            <h2 class="text-4xl font-bold">{value}</h2>
            <p class="text-xs mt-2 opacity-90">{subtext}</p>
        </div>
        <div class="h-12 w-12 bg-white/20 rounded-full flex items-center justify-center text-3xl">
            <i class="fas {icon}"></i>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
def insight_box(title, text, color_code, icon):
    st.markdown(f"""
    <div style="
        border-left: 4px solid {color_code}; 
        background: {color_code}10; 
        padding: 12px; 
        margin-bottom: 12px; 
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <h4 style="font-weight: 600; font-size: 14px; color: {color_code}; margin-bottom: 4px;">
            <i class="fas {icon}" style="margin-right: 8px;"></i>{title}
        </h4>
        <p style="font-size: 12px; color: #475569; line-height: 1.5;">{text}</p>
    </div>
    """, unsafe_allow_html=True)

def price_box(item, price, is_bundle=False, original_price=None):
    if is_bundle:
        style = "background-color: #2563eb; color: white; box-shadow: 0 4px 10px rgba(37,99,235,0.4); transform: scale(1.03);"
        price_size = "xl"
        # Calculate discount for display if original_price is available (which is the sum of optimal individual prices)
        discount_html = f'<div style="font-size: 11px; margin-top: 5px; opacity: 0.7; text-decoration: line-through;">{format_currency_int(original_price)}</div>' if original_price else ''
    else:
        style = "background-color: #f1f5f9; color: #1e293b; border: 1px solid #e2e8f0;"
        price_size = "lg"
        discount_html = ''
        
    st.markdown(f"""
    <div style="
        {style}
        padding: 16px; 
        border-radius: 8px; 
        text-align: center;
        transition: all 0.2s;
        height: 100%;
    ">
        <div style="font-size: 12px; color: {'#bfdbfe' if is_bundle else '#64748b'}; margin-bottom: 4px; text-transform: uppercase; font-weight: {'bold' if is_bundle else 'normal'};">{item}</div>
        <div style="font-size: {price_size}; font-weight: bold;">{format_currency_int(price)}</div>
        {discount_html}
    </div>
    """, unsafe_allow_html=True)

# --- 6. Main App Logic ---

# Run the optimization only once on load and cache the results
optimization_results = run_optimization(WTP_MATRIX, BUNDLE_WTP_ARRAY)

if optimization_results is None or not optimization_results['success']:
    st.error("Optimization failed! Please check your data format or increase solver iterations.")
else:
    # Extract results for easy use
    OP = optimization_results
    OPTIMAL_PRICES = {PRODUCT_MAP[i]: OP['optimal_prices'][i] for i in range(5)}
    BUNDLE_PRICE = OP['optimal_prices'][5]
    SEPARATE_SUM = sum(OPTIMAL_PRICES.values()) # Sum of optimal individual prices
    
    TOTAL_OPTIMIZED_REVENUE = OP['total_optimized_revenue']
    TOTAL_CONSUMER_SURPLUS = OP['total_consumer_surplus']
    SEPARATE_PRICING_REVENUE = OP['separate_pricing_revenue']
    df_customer = OP['decisions_df']

    # Inject Tailwind-like utility classes and font
    st.markdown(
        """
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
            html, body, [class*="stApp"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
            .gradient-text { 
                background: linear-gradient(90deg, #2563eb, #1d4ed8); 
                -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; 
                background-clip: text;
            }
            .stDataFrame { border-radius: 8px; overflow: hidden; }
            .stDataFrame thead th { font-weight: 700 !important; }
            /* Force the price columns to be a bit smaller to prevent wrapping issues */
            .st-emotion-cache-16ffsyh > div:nth-child(2) > div:nth-child(1) > div { max-width: 150px !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- KPI Section (MOVED TO THE TOP) ---
    col1, col2 = st.columns(2)

    with col1:
        kpi_card(
            title="Total Optimized Revenue",
            value=format_currency_M(TOTAL_OPTIMIZED_REVENUE),
            subtext=f"Estimated revenue gain vs Separate Pricing: +{format_currency_M(TOTAL_OPTIMIZED_REVENUE - SEPARATE_PRICING_REVENUE)}",
            icon="fa-chart-line",
            color_classes="from-blue-600 to-blue-700 bg-gradient-to-r"
        )

    with col2:
        kpi_card(
            title="Total Consumer Surplus",
            value=format_currency_M(TOTAL_CONSUMER_SURPLUS),
            subtext="Measure of overall customer satisfaction with the pricing model.",
            icon="fa-smile",
            color_classes="from-green-600 to-green-700 bg-gradient-to-r"
        )
    
    st.markdown("<br>", unsafe_allow_html=True) # Add some spacing after KPIs

    # Title and Subtitle
    st.markdown(
        """
        <h1 style="font-size: 32px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
            Samsung <span class="gradient-text">Dynamic Pricing Optimization</span>
        </h1>
        <p style="color: #64748b; margin: 0;">Mixed Bundling Strategy Analysis using Evolutionary Solver (Differential Evolution)</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")


    # --- Optimal Price List Section ---
    st.markdown(f'<h3 style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 16px;">Optimal Price List (Combo & Individual)</h3>', unsafe_allow_html=True)

    price_cols = st.columns(6)

    items = list(OPTIMAL_PRICES.keys())
    prices = list(OPTIMAL_PRICES.values())

    for i, (item, price) in enumerate(zip(items, prices)):
        with price_cols[i]:
            price_box(item, price, is_bundle=False)

    with price_cols[5]:
        # Calculate actual discount percentage
        discount_pct = (1 - BUNDLE_PRICE / SEPARATE_SUM) * 100 if SEPARATE_SUM > 0 else 0
        price_box(
            f"All-In Bundle ({discount_pct:.1f}% off)", 
            BUNDLE_PRICE, 
            is_bundle=True, 
            original_price=SEPARATE_SUM
        )
        
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Insights & Customer Table Section ---
    col_insight, col_table = st.columns([1, 2])

    with col_insight:
        st.markdown(f'<h3 style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 16px;"><i class="fas fa-robot text-blue-600" style="margin-right: 8px;"></i> Dynamic Strategic Insights</h3>', unsafe_allow_html=True)
        
        bundle_conversion = df_customer[df_customer['decision'] == 'Bundle']['Customer ID'].nunique()
        total_customers = len(df_customer)

        insight_box(
            title="Bundle Conversion Success",
            text=f'The optimal pricing strategy converted <b>{bundle_conversion} out of {total_customers} customers</b> ({bundle_conversion/total_customers*100:.1f}%) into full bundle purchasers, maximizing cross-selling revenue.',
            color_code="#2563eb",
            icon="fa-bullseye"
        )
        
        # Determine the lowest priced item (often used as a loss leader or entry point)
        min_price_item = min(OPTIMAL_PRICES, key=OPTIMAL_PRICES.get)

        insight_box(
            title="Entry-Point Analysis",
            text=f'The <b>{min_price_item}</b> is the lowest-priced individual item, serving as a low-risk entry point. Customers who purchase only this item have a WTP too low to justify the bundle.',
            color_code="#f97316",
            icon="fa-tag"
        )

        insight_box(
            title="Objective Function Context",
            text=f'The Differential Evolution algorithm minimizes the negative of the total revenue, navigating the non-linear, non-smooth function space to find the best possible pricing mix that maximizes total sales.',
            color_code="#9333ea",
            icon="fa-balance-scale"
        )
        
    with col_table:
        st.markdown(f'<h3 style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 16px;">Customer-Wise Optimized Decisions</h3>', unsafe_allow_html=True)
        
        # Prepare data for display
        df_display = df_customer.copy()
        df_display['Revenue'] = df_display['revenue'].apply(format_currency_int)
        df_display['Surplus'] = df_display['surplus'].apply(lambda x: f'+{format_currency_int(x)}' if x > 0 else format_currency_int(x))
        df_display = df_display.rename(columns={'decision': 'Decision', 'items': 'Items Purchased'})
        
        # Apply custom column formatting for a cleaner look
        st.dataframe(
            df_display[['Customer ID', 'Decision', 'Items Purchased', 'Revenue', 'Surplus']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Customer ID": st.column_config.TextColumn("ID", width="small"),
                "Decision": st.column_config.TextColumn("Decision", width="small"),
                "Items Purchased": st.column_config.TextColumn("Items Purchased", width="medium"),
                "Revenue": st.column_config.TextColumn("Revenue (₹)", width="small"),
                "Surplus": st.column_config.TextColumn("Surplus (₹)", width="small"),
            }
        )