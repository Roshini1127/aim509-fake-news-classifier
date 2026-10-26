                    st.info(f"Prediction: {prediction}")
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([cleaned])[0]
                    classes = getattr(model, "classes_", [])
                    confidence = float(max(probabilities)) * 100
                    st.metric("Highest model probability", f"{confidence:.1f}%")
                    st.caption("This is model confidence, not a calibrated probability that the news is true.")
                    st.write("Class scores")
                    score_df = pd.DataFrame({
                        "Class": [normalise_label(c) for c in classes],
                        "Model probability": [float(p) for p in probabilities]
                    }).sort_values("Model probability", ascending=False)
                    st.dataframe(score_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("This model does not provide probability estimates.")
                terms, message = get_feature_explanation(model, cleaned)
                with st.expander("Explain influential terms"):
                    if terms:
                        st.write("These are influential active features for the model, not proof or a causal explanation.")
                        st.dataframe(
                            pd.DataFrame(terms, columns=["Feature", "Model weight"]),
                            use_container_width=True, hide_index=True
                        )
                    else:
                        st.info(message or "No feature explanation is available.")
                save_history(cleaned, prediction)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_batch:
    st.subheader("Classify multiple items from a CSV file")
    st.write("Upload a CSV with a column named `text`, `headline`, or `content`.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            text_col = next((c for c in ["text", "headline", "content"] if c in df.columns), None)
            if text_col is None:
                st.error("No text column found. Rename a column to `text`, `headline`, or `content`.")
            else:
                st.write("Preview")
                st.dataframe(df.head(10), use_container_width=True)
                if st.button("Classify CSV"):
                    inputs = df[text_col].fillna("").astype(str).tolist()
                    predictions = []
                    for item in inputs:
                        if len(item.strip()) < 1:
                            predictions.append("EMPTY INPUT")
                        else:
                            predictions.append(normalise_label(model.predict([item])[0]))
                    result = df.copy()
                    result["predicted_label"] = predictions
                    st.dataframe(result, use_container_width=True)
                    st.download_button(
                        "Download classified CSV",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="classified_news.csv",
                        mime="text/csv"
                    )
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

with tab_eval:
    st.subheader("Model evaluation")
    st.caption("Metrics must come from a held-out test set. Do not report demonstration or training-set scores as final results.")
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            cols = st.columns(4)
            for col, key, label in zip(
                cols, ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
            ):
                value = metrics.get(key)
                col.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")
            st.write(f"Dataset: {metrics.get('dataset_name', 'Not specified')}")
            st.write(f"Test samples: {metrics.get('test_samples', 'Not specified')}")
            st.write(f"Label column: {metrics.get('label_column', 'Not specified')}")
            if "confusion_matrix" in metrics and "class_names" in metrics:
                cm = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["class_names"],
                    columns=metrics["class_names"]
                )
                st.write("Confusion matrix")
                st.dataframe(cm, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read evaluation metrics: {exc}")
    else:
        st.info("No model_metrics.json found yet. Train the model with a labelled dataset using `python train_model.py`.")
    if COMPARISON_PATH.exists():
        st.write("Model comparison")
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if "f1_macro" in comparison.columns and len(comparison) > 0:
            fig, ax = plt.subplots()
            ax.bar(comparison["model"], comparison["f1_macro"])
            ax.set_ylabel("Macro F1")
            ax.set_title("Model comparison (held-out test set)")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
            plt.close(fig)

with tab_history:
    st.subheader("Recent prediction history")
    st.caption("History is stored locally in SQLite on the running app environment. Cloud storage may reset or be ephemeral.")
    ensure_history_db()
    with sqlite3.connect(HISTORY_DB) as conn:
        history = pd.read_sql_query(
            "SELECT created_at, text_preview, prediction FROM history ORDER BY id DESC LIMIT 100", conn
        )
    if history.empty:
        st.info("No predictions recorded in this app session yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            with sqlite3.connect(HISTORY_DB) as conn:
                conn.execute("DELETE FROM history")
            st.rerun()

with tab_about:
    st.subheader("Project scope")
    st.markdown("""
    - **Text representation:** word and character TF-IDF features.
    - **Model comparison:** Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Classifier.
    - **Explainability:** influential active terms for compatible linear models.
    - **Batch processing:** CSV upload and downloadable predictions.
    - **Evaluation:** held-out test metrics and confusion matrix.
    - **Limitations:** dataset bias, changing language, satire, missing context, and adversarial text can cause errors.
    """)
    st.write("Project course: AIM-509 Project Work")
