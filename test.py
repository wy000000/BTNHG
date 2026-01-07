def _save_epoch_loss_list(self):
	fileName = f"{int(time.time() * 1000) % 1000:03d} " \
		+ BTNHGV2ParameterClass.epoch_loss_listFileName
	filePath = os.path.join(self.methodFolderPath, fileName)

	epoch_loss_list = []
	if self.model.epoch_loss_list is not None:
		epoch_loss_list = self.model.epoch_loss_list

	# 创建 DataFrame
	df = pd.DataFrame(epoch_loss_list, columns=['Epoch', 'Loss', 'Accuracy'])

	# 保存到 Excel，并生成散点图
	with pd.ExcelWriter(filePath, engine="xlsxwriter") as writer:
		df.to_excel(writer, sheet_name="Sheet1", index=False)

		workbook  = writer.book
		worksheet = writer.sheets["Sheet1"]

		# 创建散点图（带平滑线）
		chart = workbook.add_chart({"type": "scatter", "subtype": "smooth"})

		# Loss 曲线
		chart.add_series({
			"name":       "Loss Curve",
			"categories": ["Sheet1", 1, 0, len(df), 0],  # Epoch 列
			"values":     ["Sheet1", 1, 1, len(df), 1],  # Loss 列
			"marker":     {"type": "circle"},
			"smooth":     True
		})

		# Accuracy 曲线
		chart.add_series({
			"name":       "Accuracy Curve",
			"categories": ["Sheet1", 1, 0, len(df), 0],  # Epoch 列
			"values":     ["Sheet1", 1, 2, len(df), 2],  # Accuracy 列
			"marker":     {"type": "square"},
			"smooth":     True
		})

		chart.set_title({"name": "Epoch vs Loss & Accuracy"})
		chart.set_x_axis({"name": "Epoch"})
		chart.set_y_axis({"name": "Value"})  # Y轴统一为数值

		# 插入图表到 Excel
		worksheet.insert_chart("D2", chart)

	print(f"{fileName}已保存")
	return filePath
