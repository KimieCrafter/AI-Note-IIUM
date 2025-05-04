class ActivityClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Activity Classifier")
        self.root.geometry("800x700")
        
        # Try to load the model
        try:
            self.model = joblib.load('best_model.pkl')
            print("Model loaded successfully!")
            # Check if the model supports prediction probabilities
            self.has_proba = hasattr(self.model, 'predict_proba')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.model = None
            self.has_proba = False

        # Activity labels - update these to match your model's classes
        self.activity_labels = ['STANDING', 'LAYING', 'WALKING', 'SITTING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        # Try to extract class labels from model if available
        if self.model is not None and hasattr(self.model, 'classes_'):
            try:
                classes = self.model.classes_
                if len(classes) > 0:
                    self.activity_labels = [str(c) for c in classes]
                    print(f"Loaded class labels from model: {self.activity_labels}")
            except:
                print("Could not extract class labels from model, using defaults")

        self.label = tk.Label(root, text="Upload an Excel file with activity features", font=("Arial", 14))
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Excel File", command=self.upload_file)
        self.upload_button.pack(pady=5)

        # Result label
        self.result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
        self.result_label.pack(pady=5)
        
        # Confidence label
        self.confidence_label = tk.Label(root, text="", font=("Arial", 11), fg="green")
        self.confidence_label.pack(pady=2)

        # Chart type selection
        self.chart_type = tk.StringVar(value="Pie")
        chart_type_frame = tk.Frame(root)
        chart_type_frame.pack(pady=5)
        tk.Label(chart_type_frame, text="Chart Type:").pack(side="left", padx=5)
        tk.Radiobutton(chart_type_frame, text="Pie Chart", variable=self.chart_type, value="Pie", command=self.update_chart).pack(side="left")
        tk.Radiobutton(chart_type_frame, text="Bar Chart", variable=self.chart_type, value="Bar", command=self.update_chart).pack(side="left")

        # Sample selector
        self.sample_frame = tk.Frame(root)
        self.sample_frame.pack(pady=5)
        tk.Label(self.sample_frame, text="Sample:").pack(side="left", padx=5)
        self.sample_selector = ttk.Combobox(self.sample_frame, width=10)
        self.sample_selector.pack(side="left")
        self.sample_selector.bind("<<ComboboxSelected>>", self.on_sample_change)

        # Chart display frame
        self.chart_frame = tk.Frame(root)
        self.chart_frame.pack(fill='both', expand=True, pady=10)
        self.canvas = None

        # Data table frame
        self.table_frame = tk.Frame(root)
        self.table_frame.pack(fill='both', expand=True, pady=10)
        
        # Store the current sample index
        self.current_sample = 0
        
        # Store confidence scores
        self.confidence_scores = None

    def on_sample_change(self, event):
        try:
            self.current_sample = int(self.sample_selector.get()) - 1
            self.update_display()
        except:
            pass

    def update_chart(self):
        self.update_display()

    def update_display(self):
        if hasattr(self, 'df') and self.confidence_scores is not None:
            # Update the prediction label
            prediction = self.predictions[self.current_sample]
            self.result_label.config(text=f"🧍 The user is most likely: {prediction.upper()}")
            
            # Update confidence label
            confidence = self.confidence_scores[self.current_sample]
            max_conf = confidence.max() * 100
            self.confidence_label.config(text=f"Confidence: {max_conf:.1f}%")
            
            # Redisplay the chart
            self.display_chart()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not file_path:
            return
            
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Cannot make predictions.")
            return

        try:
            self.df = pd.read_excel(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")
            return

        try:
            # Prepare features - remove non-feature columns
            feature_columns = self.df.columns.tolist()
            columns_to_drop = []
            
            potential_non_features = ['activity_label', 'label', 'activity', 'id', 'timestamp', 'subject', 'user_id']
            for col in potential_non_features:
                if col in feature_columns:
                    columns_to_drop.append(col)
            
            features = self.df.drop(columns_to_drop, axis=1, errors='ignore')
            
            # Ensure all features are numeric
            for col in features.columns:
                if not pd.api.types.is_numeric_dtype(features[col]):
                    try:
                        features[col] = pd.to_numeric(features[col])
                    except:
                        features = features.drop(col, axis=1)
                        print(f"Dropped non-numeric column: {col}")
            
            # Get predictions
            self.predictions = self.model.predict(features)
            
            # Get confidence scores
            if self.has_proba:
                self.confidence_scores = self.model.predict_proba(features)
            else:
                # Create mock confidence scores if predict_proba not available
                num_classes = len(self.activity_labels)
                self.confidence_scores = np.zeros((len(self.predictions), num_classes))
                
                for i, pred in enumerate(self.predictions):
                    # Convert prediction to index
                    if isinstance(pred, (str, np.str_)):
                        try:
                            pred_idx = self.activity_labels.index(pred)
                        except ValueError:
                            pred_idx = 0
                    else:
                        pred_idx = int(pred) % num_classes
                    
                    # Set high confidence for predicted class, distribute rest
                    self.confidence_scores[i, pred_idx] = 0.8
                    other_classes = [j for j in range(num_classes) if j != pred_idx]
                    for j in other_classes:
                        self.confidence_scores[i, j] = 0.2 / (num_classes - 1)
            
            # Add predictions to dataframe
            self.df['Predicted Activity'] = self.predictions
            
            # Set up sample selector
            num_samples = len(self.df)
            self.sample_selector['values'] = list(range(1, num_samples + 1))
            self.sample_selector.current(0)
            self.current_sample = 0
            
            # Reset display
            self.update_display()
            self.display_table()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            return

    def display_chart(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        # Get confidence scores for the current sample
        confidence = self.confidence_scores[self.current_sample] # type: ignore
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        colors = plt.get_cmap("Set3").colors[:len(self.activity_labels)] # type: ignore
        
        if self.chart_type.get() == "Pie":
            def format_pct(pct):
                # Only show percentage if it's significant
                return f'{pct:.1f}%' if pct >= 3 else ''
            
            wedges, texts, autotexts = ax.pie( # type: ignore
                confidence,
                labels=self.activity_labels,
                autopct=format_pct,
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.4),
                textprops=dict(color="black", fontsize=9),
                pctdistance=0.85
            )
            
            # Format the labels
            for i, (wedge, autotext, activity) in enumerate(zip(wedges, autotexts, self.activity_labels)):
                # Format confidence percentage
                conf_pct = confidence[i] * 100
                
                # Adjust text color for better visibility
                if conf_pct >= 40:  # High confidence slices get white text
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                # Only show the activity label if the slice is big enough
                if conf_pct < 3:
                    texts[i].set_text('')
            
            ax.set_title(f"Confidence Scores for Sample {self.current_sample + 1}", fontsize=11)
            ax.legend(wedges, [f'{self.activity_labels[i]} ({confidence[i]:.2f})' for i in range(len(self.activity_labels))], 
                      title="Activities", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
            
        else:  # Bar chart
            # Create bar chart of confidence scores
            bars = ax.bar(self.activity_labels, confidence, color=colors)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Confidence Score")
            ax.set_xlabel("Activity")
            ax.set_title(f"Confidence Scores for Sample {self.current_sample + 1}", fontsize=11)
            
            # Add confidence values on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{confidence[i]:.2f}', ha='center', fontsize=9)

        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def display_table(self):
        # Clear existing table
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        tree = ttk.Treeview(self.table_frame)
        tree.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)

        tree["columns"] = list(self.df.columns)
        tree["show"] = "headings"

        for col in self.df.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=100)

        for _, row in self.df.iterrows():
            tree.insert("", "end", values=list(row))

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ActivityClassifierApp(root)
    root.mainloop()