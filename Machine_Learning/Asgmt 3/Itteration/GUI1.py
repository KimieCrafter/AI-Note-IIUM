class ActivityClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Activity Classifier")
        self.root.geometry("800x700")

        self.label = tk.Label(root, text="Upload an Excel file with activity features", font=("Arial", 14))
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Excel File", command=self.upload_file)
        self.upload_button.pack(pady=5)

        # Result label
        self.result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
        self.result_label.pack(pady=5)

        # Chart type selection
        self.chart_type = tk.StringVar(value="Pie")
        chart_type_frame = tk.Frame(root)
        chart_type_frame.pack(pady=5)
        tk.Label(chart_type_frame, text="Chart Type:").pack(side="left", padx=5)
        tk.Radiobutton(chart_type_frame, text="Pie Chart", variable=self.chart_type, value="Pie", command=self.update_chart).pack(side="left")
        tk.Radiobutton(chart_type_frame, text="Bar Chart", variable=self.chart_type, value="Bar", command=self.update_chart).pack(side="left")

        # Chart display frame
        self.chart_frame = tk.Frame(root)
        self.chart_frame.pack(fill='both', expand=True, pady=10)
        self.canvas = None

        # Data table frame
        self.table_frame = tk.Frame(root)
        self.table_frame.pack(fill='both', expand=True, pady=10)

    def update_chart(self):
        if hasattr(self, 'df'):
            self.display_chart(self.df)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not file_path:
            return

        try:
            self.df = pd.read_excel(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")
            return

        try:
            predictions = model.predict(self.df)
            self.df['Predicted Activity'] = predictions
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            return

        top_activity = self.df['Predicted Activity'].mode()[0]
        self.result_label.config(text=f"🧍 The user is most likely: {top_activity.upper()}")

        self.display_chart(self.df)
        self.display_table(self.df)

    def display_chart(self, df):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        counts = df['Predicted Activity'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        colors = plt.get_cmap("Set3").colors[:len(counts)] # type: ignore

        if self.chart_type.get() == "Pie":
            def autopct_format(pct):
                return ('%1.1f%%' % pct) if pct >= 5 else ''

            wedges, texts, autotexts = ax.pie( # type: ignore
                counts,
                autopct=autopct_format,
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.4),
                textprops=dict(color="black", fontsize=9),
                pctdistance=0.85
            )

            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color("black")

            ax.legend(wedges, counts.index, title="Activities", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
            ax.set_title("Predicted Activity Distribution", fontsize=11)
        else:
            ax.bar(counts.index, counts.values, color=colors)
            ax.set_ylabel("Count")
            ax.set_xlabel("Activity")
            ax.set_title("Activity Frequency", fontsize=11)
            for i, val in enumerate(counts.values):
                ax.text(i, val + 0.5, str(val), ha='center', fontsize=9)

        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def display_table(self, df):
        # Clear existing table
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        tree = ttk.Treeview(self.table_frame)
        tree.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)

        tree["columns"] = list(df.columns)
        tree["show"] = "headings"

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=100)

        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ActivityClassifierApp(root)
    root.mainloop()