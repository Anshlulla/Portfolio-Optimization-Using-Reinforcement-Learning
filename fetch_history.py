import os
from flask import Flask, jsonify, request, send_from_directory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataloader import download_price_data
from flask_cors import CORS 

app = Flask(__name__)
CORS(app, origins='*')


# Ensure the 'static/plots' directory exists for saving the images
os.makedirs('static/plots', exist_ok=True)

def plot_stock_performance(tickers: list, start: str, end: str, file_path: str) -> None:
    """
    Generate the stock performance plot (adjusted close prices) for given tickers over the specified date range
    and save it to the provided file path.
    """
    # Download stock data
    data = download_price_data(tickers, start, end)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        plt.plot(data[ticker], label=ticker)
    
    # Add labels and title
    plt.title(f"Stock Performance of {' '.join(tickers)} from {start} to {end}")
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend(loc='upper left')
    
    # Save the plot to the file path
    plt.savefig(file_path)
    plt.close()

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    tickers = request.args.get('tickers', '').split(',')
    start_date = request.args.get('start', '2023-01-01')
    end_date = request.args.get('end', '2025-01-01')
    
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    
    try:
        # Define file path to save the plot
        plot_filename = f"stock_performance_{start_date}_{end_date}.png"
        plot_file_path = os.path.join('static/plots', plot_filename)

        # Generate the plot and save it to the file path
        plot_stock_performance(tickers, start_date, end_date, plot_file_path)

        # Return the file path of the plot
        return jsonify({
            'message': 'Stock performance plot generated successfully!',
            'plot_url': f"http://localhost:8004/static/plots/{plot_filename}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    """
    Serve the plot image file from the static directory.
    """
    return send_from_directory('static/plots', filename)

if __name__ == "__main__":
    app.run(debug=True, port=8004)
