import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

os.system("cls")

dataset_path = 'C:\\Users\\Caetanogp123\\Desktop\\PROJETOS PY\\novo_venv\\data\\retail_sales_dataset.csv'

dataset = pd.read_csv(dataset_path)

dataset.fillna(0, inplace=True)

features = ['Age', 'Quantity', 'Price per Unit']
X = dataset[features]
y = dataset['Product Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

def paginate_dataframe(df, page_size=20):
    total_rows = df.shape[0]
    for start_row in range(0, total_rows, page_size):
        end_row = min(start_row + page_size, total_rows)
        print(df.iloc[start_row:end_row])
        user_input = input("Press Enter to continue or 'Q' to quit...")
        if user_input.lower() == 'q':
            break

def filter_data(dataset):
    print("Filter Options:")
    print("1. By Date Range")
    print("2. By Product Category")
    print("3. By Age Range")
    choice = input("Choose a filter option: ")
    
    if choice == '1':
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        filtered = dataset[(dataset['Date'] >= start_date) & (dataset['Date'] <= end_date)]
        paginate_dataframe(filtered)
    elif choice == '2':
        category = input("Enter product category: ")
        filtered = dataset[dataset['Product Category'] == category]
        paginate_dataframe(filtered)
    elif choice == '3':
        min_age = int(input("Enter minimum age: "))
        max_age = int(input("Enter maximum age: "))
        filtered = dataset[(dataset['Age'] >= min_age) & (dataset['Age'] <= max_age)]
        paginate_dataframe(filtered)
    else:
        print("Invalid option")

def search_data(dataset):
    search_term = input("Enter search term: ")
    results = dataset[dataset.apply(lambda row: row.astype(str).str.contains(search_term).any(), axis=1)]
    paginate_dataframe(results)

while True:
    print("Options Menu:")
    print("1. View the first lines of the dataset")
    print("2. General statistics")
    print("3. Sales chart by category")
    print("4. Purchases by gender")
    print("5. Age group distribution")
    print("6. Monthly revenue")
    print("7. Exit")
    print("8. Predict product category")
    print("9. View most expensive and cheapest transactions")
    print("10. Filter data")
    print("11. Search data")
    
    choice = input("Choose an option: ")
    print()
    
    if choice == '1':
        print("First lines of the dataset:")
        print(dataset.head(10))
        print()
        
    elif choice == '2':
        print("General statistics of the dataset:")
        print("Statistics for numeric columns:")
        print(dataset.describe())
        print()
        print("Statistics for categorical columns:")
        print(dataset.describe(include=[object]))
        print()
        
    elif choice == '3':
        if 'Product Category' in dataset.columns:
            print("Sales by Product Category:")
            print(dataset['Product Category'].value_counts())
            dataset['Product Category'].value_counts().plot(kind='bar', color='skyblue', title='Sales by Product Category')
            plt.xlabel("Categories")
            plt.ylabel("Number of sales")
            plt.show()
        else:
            print("The 'Product Category' column was not found in the dataset.")
        print()
    
    elif choice == '4':
        if 'Gender' in dataset.columns:
            print("Purchases by Gender:")
            print(dataset['Gender'].value_counts())
            dataset['Gender'].value_counts().plot(kind='bar', color='lightgreen', title='Purchases by Gender')
            plt.xlabel("Gender")
            plt.ylabel("Number of Purchases")
            plt.show()
        else:
            print("The 'Gender' column was not found in the dataset.")
        print()
    
    elif choice == '5':
        if 'Age' in dataset.columns:
            bins = [18, 25, 35, 45, 55, 65, 75, 85, 95]
            labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95']
            dataset['Age Group'] = pd.cut(dataset['Age'], bins=bins, labels=labels)
            print("Age Group Distribution:")
            print(dataset['Age Group'].value_counts().sort_index())
            dataset['Age Group'].value_counts().sort_index().plot(kind='bar', color='orange', title='Age Group Distribution')
            plt.xlabel("Age Group")
            plt.ylabel("Number of Customers")
            plt.show()
        else:
            print("The 'Age' column was not found in the dataset.")
        print()
    
    elif choice == '6':
        if 'Date' in dataset.columns:
            dataset['Date'] = pd.to_datetime(dataset['Date'])
            dataset.set_index('Date', inplace=True)
            monthly_sales = dataset.resample('M')['Total Amount'].sum()
            print("Monthly Revenue:")
            print(monthly_sales)
            monthly_sales.plot(kind='line', title='Monthly Revenue', marker='o')
            plt.xlabel("Month")
            plt.ylabel("Total Revenue")
            plt.show()
        else:
            print("The 'Date' column was not found in the dataset.")
        print()
    
    elif choice == '7':
        print("Exiting the program. Goodbye!")
        break
    
    elif choice == '8':
        print("Model Evaluation:")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        print(f"Features used for prediction: {features}")
        sample_data = pd.DataFrame([[35, 2, 100]], columns=features)
        prediction = model.predict(sample_data)
        print(f"Predicted Product Category for sample data {sample_data.values.tolist()}: {prediction[0]}")
        sample_accuracy = model.score(X_test, y_test) * 100
        print(f"Sample Prediction Accuracy: {sample_accuracy:.2f}%")
        print()
    
    elif choice == '9':
        most_expensive = dataset.loc[dataset['Total Amount'].idxmax()]
        cheapest = dataset.loc[dataset['Total Amount'].idxmin()]
        print("Most Expensive Transaction:")
        print(most_expensive)
        print("\nCheapest Transaction:")
        print(cheapest)
        print()
    
    elif choice == '10':
        filter_data(dataset)
        print()
    
    elif choice == '11':
        search_data(dataset)
        print()
    
    else:
        print("Invalid choice. Please try again.")
        print()
