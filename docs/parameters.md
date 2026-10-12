当前正式训练配置（D768 / 12 layers / 1-bin / dual event camera）的参数量为：
模块	参数量	是否训练
DINOv3 ViT-L/16	303,129,600	冻结
T5-small encoder	35,330,816	冻结
Event expert（codec、12 层 Event branch、相机嵌入）	193,776,646	训练
Action expert（输入/输出头、12 层 Action branch）	171,319,303	训练
Context adapters（DINO/T5 投影、RGB 相机与模态嵌入）	1,184,256	训练
总计	704,740,621	
实际优化器更新	366,280,205	
冻结参数	338,460,416	


也就是总模型约 704.74M 参数，其中约 366.28M 是当前训练会更新的参数。
