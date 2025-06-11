import os
import random
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from flask_cors import CORS
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
CORS(app)

# Helper function to generate portfolio allocation using random weights
def generate_portfolio_allocation(companies_count: int, total_capital: float):
    # Generate random weights for each company (sum of weights = 1)
    weights = [random.random() for _ in range(companies_count)]
    weight_sum = sum(weights)
    
    # Normalize weights to sum to 1
    normalized_weights = [w / weight_sum for w in weights]
    
    # Allocate capital based on normalized weights
    allocation = [total_capital * w for w in normalized_weights]
    return allocation, normalized_weights  # Return both the allocation and the weights

# Endpoint to get portfolio allocation and generate pie chart
@app.route('/portfolio_allocation', methods=['GET'])
def portfolio_allocation():
    companies_count = int(request.args.get('companies', 3))
    total_capital = float(request.args.get('capital', 100000))  # Default to 100,000

    if companies_count not in [3, 5, 10]:
        return jsonify({"error": "Invalid number of companies. Choose 3, 5, or 10."}), 400
    
    try:
        # Generate portfolio allocation using random weights
        allocation, weights = generate_portfolio_allocation(companies_count, total_capital)
        
        # Prepare pie chart
        companies = [f"Company {i+1}" for i in range(companies_count)]
        plt.figure(figsize=(8, 8))
        plt.pie(allocation, labels=companies, autopct='%1.1f%%', startangle=140)
        plt.title(f"Capital Allocation for {companies_count} Companies")
        
        # Save plot as a file
        plot_filename = f"portfolio_allocation_{companies_count}_{total_capital}.png"
        plot_file_path = os.path.join('static/plots', plot_filename)
        plt.savefig(plot_file_path)
        plt.close()
        
        # Return plot URL and allocation details
        return jsonify({
            'message': 'Portfolio allocation pie chart generated successfully!',
            'plot_url': f"http://localhost:8001/static/plots/{plot_filename}",
            'allocation': allocation,
            'weights': weights
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory('static/plots', filename)

if __name__ == "__main__":
    app.run(debug=True, port=8001)
