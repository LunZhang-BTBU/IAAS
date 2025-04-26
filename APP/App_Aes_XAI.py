
import os
import random
from scipy import stats
from scipy.stats import rankdata

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from omegaconf import OmegaConf

from torch_aesthetics.models import *
from torch_aesthetics.cluster import *
from torch_aesthetics.cluster_app import *
from torch_aesthetics.losses import *
from torch_aesthetics.aadb import AADB, load_transforms
from torch_aesthetics.kan_figure import *


class ImageDatasetApp:
    def __init__(self, root):


        self.root = root
        self.root.title("IAAS")
        self.root.geometry("1200x800")

         # 颜色配置
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'background': '#ecf0f1',
            'text': '#2c3e50'
        }
        

        # 创建按钮样式
        style = ttk.Style()
        style.configure('Bold.TButton', font=('Arial', 12, 'bold'))
        # 字体配置
        self.fonts = {
            'title': ('Segoe UI', 16, 'bold'),
            'body': ('Segoe UI', 16),
            'button': ('Segoe UI', 16, 'bold')
        }
        
        # 主容器布局
        self._setup_main_panes()

        # 初始化变量
        self.dataset = None
        self.dataloader = None
        self.current_path = ""
        
        # 界面布局
        # self.create_widgets()
        # self.setup_image_grid()

         # ...原有初始化代码...
        self.cluster_results = None
        # self._add_cluster_controls()  # 添加聚类控制
        # 新增评分相关属性
        self.scoring_model = None
        self.scores = []
        self._add_scoring_controls()  # 添加评分控制

        # 新增聚类评分相关属性
        self.cluster_scores = {}  # 存储各聚类组的评分数据
        self._add_cluster_score_controls()
        self.cluster_app = None

        self.current_cluster_id = -1  # 新增：当前选中聚类ID
        self.sorted_index_list = []    # 新增：排序后的全局索引列表
        self.clusterID_get_indices = {}

        # 修改标签页初始化
        self._init_analysis_tab()

        self.heatmap_model = None  # 新增热力图模型引用
        self.heatmap_cache = {}    # 新增热力图缓存

        self.analysis_queue = queue.Queue()  # 新增消息队列
        self.root.after(100, self.start_queue_polling)  # 启动队列轮询


        self.analysis_texts = {}  # 新增字典存储各图片分析框
        self.current_active_idx = -1  # 追踪当前显示索引
        
    # def _trigger_analysis(self, global_idx):
    #     """触发图片分析"""
    #     # 清空旧内容并显示加载状态
    #     self.analysis_text.delete(1.0, tk.END)
    #     self.analysis_text.insert(tk.END, "分析中，请稍候...")
    #     self.analysis_text.update()
        
    #     # 在后台线程执行分析
    #     threading.Thread(
    #         target=self._async_analyze_image,
    #         args=(global_idx,),
    #         daemon=True
    #     ).start()



    def _async_analyze_image(self, global_idx):
        """带索引的异步分析"""
        try:
            image_path = self.dataset.image_paths[global_idx]
            result = self.analyze_image(image_path)
            self.analysis_queue.put((global_idx, result))  # 修改为发送元组
        except Exception as e:
            error_msg = f"分析失败: {str(e)}"
            self.analysis_queue.put((global_idx, error_msg))

    def _update_analysis_result(self, global_idx, result):
        """安全更新指定文本框"""
        if global_idx != self.current_active_idx:
            return  # 防止显示错位
        
        text_widget = self.analysis_texts.get(global_idx)
        if text_widget:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, result)
            text_widget.see(tk.END)
            text_widget.config(state=tk.DISABLED)

    def start_queue_polling(self):
        """修改后的队列轮询"""
        try:
            while True:
                global_idx, result = self.analysis_queue.get_nowait()
                self._update_analysis_result(global_idx, result)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.start_queue_polling)



    
    def analyze_image(self, image_path):
        """Kimi API分析实现"""
        # 创建 OpenAI 客户端实例
        client = OpenAI(
        )

        # 图片编码处理
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:image/{os.path.splitext(image_path)[1][1:]};base64,{image_base64}"

        # 构建消息
        messages = [
            {"role": "system", "content": "You are a professional photography critic, analyzing image aesthetics from composition, color, lighting, theme, and balance"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Please analyze the aesthetic features of this image"}
                ]
            }
        ]

        # 发送请求
        response = client.chat.completions.create(
            model="moonshot-v1-8k-vision-preview",
            messages=messages
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content



    def _init_analysis_tab(self):
        """初始化分析标签页"""
        # 清空旧内容
        for widget in self.aesth_tab.winfo_children():
            widget.destroy()
        
        # 创建固定滚动系统
        self.analysis_canvas = tk.Canvas(self.aesth_tab, bg=self.colors['background'])
        scroll_y = ttk.Scrollbar(self.aesth_tab, orient=tk.VERTICAL, command=self.analysis_canvas.yview)
        scroll_x = ttk.Scrollbar(self.aesth_tab, orient=tk.HORIZONTAL, command=self.analysis_canvas.xview)
        
        self.analysis_canvas.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )
        
        # 布局滚动条
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建内容容器
        self.analysis_container = ttk.Frame(self.analysis_canvas)
        self.analysis_canvas.create_window((0,0), window=self.analysis_container, anchor=tk.NW)
        self.analysis_container.bind("<Configure>", 
            lambda e: self.analysis_canvas.configure(scrollregion=self.analysis_canvas.bbox("all")))

    def _add_cluster_score_controls(self):
        """Add cluster scoring control components"""
        score_frame = ttk.LabelFrame(self.control_panel, text="Cluster Scoring")
        score_frame.pack(pady=10, padx=5, fill=tk.X)

        # Statistical metric selection
        self.metric_var = tk.StringVar(value="mean")
        metrics = [("Mean", "mean"), ("Maximum", "max"), ("Minimum", "min")]
        for text, val in metrics:
            ttk.Radiobutton(
                score_frame,
                text=text,
                variable=self.metric_var,
                value=val
            ).pack(side=tk.LEFT, padx=2)

        # Refresh button
        ttk.Button(
            score_frame,
            text="Refresh Statistics",
            command=self.update_cluster_stats,
            style='Accent.TButton'
        ).pack(side=tk.RIGHT)
        
    def _add_scoring_controls(self):
        """Add scoring control components"""
        scoring_frame = ttk.LabelFrame(self.control_panel, text="Image Scoring")
        scoring_frame.pack(pady=10, padx=5, fill=tk.X)

        self.scoring_btn = ttk.Button(
            scoring_frame,
            text="⭐ Generate Score",
            command=self.run_scoring,
            style='Bold.TButton'
        )
        self.scoring_btn.pack(fill=tk.X, pady=3)

        # Score display settings
        self.show_score_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            scoring_frame,
            text="Show Detailed Attributes",
            variable=self.show_score_var,
            command=self.toggle_score_display
        ).pack(anchor=tk.W)



    def run_scoring(self):
        """执行评分操作"""
        if not self.dataset:
            messagebox.showwarning("警告", "请先加载数据集！")
            return

        if not self._load_scoring_model():
            return

        # 禁用按钮防止重复点击
        self.scoring_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0

        # 在后台线程执行评分
        threading.Thread(
            target=self._perform_scoring,
            daemon=True
        ).start()
    
    def _load_scoring_model(self):
        """加载评分模型"""
        try:
            if self.scoring_model is None:
                path_Reg = '/home/zl/下载/input/pykan-master/models/Cam_Lin_reg/Cam_Lin_reg_res50_y_12_epoch_13_loss_0.0696_grid_1_score_0.5755565230299889.pt'
                cfg = OmegaConf.load("configs/train.yaml")
                
                self.scoring_model = RegressionNetwork_kan(
                    backbone='resnet50',
                    num_attributes=12,
                    pretrained=cfg.models.pretrained,
                    kan=None,
                )
                self.scoring_model.load_state_dict(torch.load(path_Reg))
                self.scoring_model.to(cfg.device).float().eval()
            return True
        except Exception as e:
            messagebox.showerror("错误", f"加载评分模型失败：{str(e)}")
            return False
    


     # 修改原评分方法以支持聚类
    def _perform_scoring(self):
        """执行评分并关联聚类索引"""
        try:
            cfg = OmegaConf.load("configs/train.yaml")
            self.scores = []
            
            with torch.no_grad():
                # 改为使用数据加载器批量处理
                for batch in self.dataloader:
                    inputs = batch.to(cfg.device)
                    outputs = self.scoring_model(inputs).cpu().numpy()

                    print(outputs.shape)
                    
                    # 转换每个batch的评分
                    attributes = ['Aesth_score', 'balancing_ele', 'color_harmony',
                                'content', 'depth_of_field', 'light', 'motion_blur',
                                'object', 'repetition', 'rule_of_thirds', 'symmetry',
                                'vivid_color']

                    # 处理三维输出结构
                    if outputs.ndim == 3:
                        # 去除批次维度并转换为二维数组
                        outputs = outputs.squeeze(0)  # 形状变为 (16, 12)
                    elif outputs.ndim != 2:
                        raise ValueError(f"无效的输出维度：{outputs.ndim}，预期2D或3D数组")

                    # 转换为Python原生float类型
                    processed_outputs = []
                    for sample in outputs:
                        if hasattr(sample, 'cpu'):  # 处理PyTorch张量
                            sample = sample.cpu().detach().numpy()
                        if hasattr(sample, 'astype'):  # 处理numpy数组
                            sample = sample.astype(float)
                        processed_outputs.append([float(x) for x in sample])

                    # 构建评分字典
                    self.scores.extend(
                        [dict(zip(attributes, sample)) 
                        for sample in processed_outputs]
                    )

                    # print(f"成功添加 {len(processed_outputs)} 个样本评分")
                    # print("首个样本评分示例：", self.scores[0])


                    
                    # 更新进度
                    progress = len(self.scores) / len(self.dataset) * 100
                    self.root.after(0, lambda v=progress: self.progress.config(value=v))
            
            # 关联聚类索引
            if hasattr(self.cluster_app, 'indices'):
                self._analyze_cluster_scores()
            
            self.root.after(0, self._update_display_with_scores)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"评分失败：{str(e)}"))

    def _analyze_cluster_scores(self):
        """分析各聚类组的评分特征"""
        self.cluster_scores = {}
        
        for cluster_id in range(int(self.cluster_num.get())):
            indices = self.cluster_app.indices[cluster_id]
            valid_indices = [i for i in indices if i < len(self.scores)]
            
            # 收集所有属性数据
            cluster_data = {
                attr: [] for attr in self.scores[0].keys()
            }
            
            for idx in valid_indices:
                for attr, value in self.scores[idx].items():
                    cluster_data[attr].append(value)
            
            # 计算统计量
            self.cluster_scores[cluster_id] = {
                attr: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
                for attr, values in cluster_data.items()
            }

    def show_cluster_analysis(self):
        """显示聚类分析报告"""
        if not self.cluster_scores:
            messagebox.showinfo("提示", "请先完成聚类和评分")
            return
        
        analysis_win = tk.Toplevel()
        analysis_win.title("聚类分析报告")
        
        # 创建表格
        columns = ['属性'] + [f"类别 {i}" for i in self.cluster_scores.keys()]
        tree = ttk.Treeview(analysis_win, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # 填充数据
        for attr in self.scores[0].keys():
            row = [attr]
            for cluster_id in self.cluster_scores:
                stats = self.cluster_scores[cluster_id][attr]
                row.append(f"{stats['mean']:.2f} ± {stats['std']:.2f}")
            tree.insert("", tk.END, values=row)
        
        tree.pack(fill=tk.BOTH, expand=True)

    def _update_display_with_scores(self):
        """更新带评分的显示"""
        self.show_dataset()

    def toggle_score_display(self):
        """切换评分显示模式"""
        if hasattr(self, 'scores'):
            self.show_dataset()

    def update_cluster_stats(self):
        """更新聚类统计信息"""
        if self.cluster_app is not None:
            self._show_cluster_results()
    
    
    
    



    def _setup_main_panes(self):
        """创建主布局面板"""
        # 使用PanedWindow实现可调整的分割布局
        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.main_pane.pack(fill=tk.BOTH, expand=1)

        # 左侧控制面板
        self.control_panel = ttk.Frame(self.main_pane, width=300)
        self.main_pane.add(self.control_panel)
        
        # 右侧显示区域
        self.display_panel = ttk.Frame(self.main_pane)
        self.main_pane.add(self.display_panel, minsize=700)

        # 构建子组件
        self._build_control_panel()
        self._build_display_panel()
        self._build_status_bar()
    
    def _build_control_panel(self):
        """Build the left control panel"""
        # Panel title
        title_frame = ttk.Frame(self.control_panel)
        title_frame.pack(pady=10, fill=tk.X)
        ttk.Label(title_frame, text="Control Panel", font=self.fonts['title'], 
                foreground=self.colors['primary']).pack()

        # Function navigation
        self._build_navigation()
        
        # Clustering controls
        self._add_cluster_controls()
        
        # # System settings
        # self._build_settings()

    def _build_display_panel(self):
        """Build the right display area"""
        # Tab component
        self.notebook = ttk.Notebook(self.display_panel)
        self.notebook.pack(fill=tk.BOTH, expand=1)
        
        # 创建标签页样式
        style = ttk.Style()
        style.configure('Tab.TNotebook.Tab', font=('Arial', 16, 'bold'))  # 增加字体大小并加粗
        
        # Aesthetic quality assessment
        self.raw_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_tab, text="Aesthetic Quality Assessment")
        
        # Aesthetic analysis tab
        self.aesth_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.aesth_tab, text="Aesthetic Attribute Analysis")
        
        # # Clustering results tab
        # self.cluster_tab = ttk.Frame(self.notebook)
        # self.notebook.add(self.cluster_tab, text="Aesthetic Attribute Analysis")
        
        # Initialize display components
        self.setup_image_grid(self.raw_tab)  # Modified parameter for setup_image_grid

    def _build_status_bar(self):
        """Build the bottom status bar"""
        self.status_bar = ttk.Frame(self.root, height=22, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(
            self.status_bar, 
            text="Ready",
            anchor=tk.W,
            font=('Segoe UI', 9)
        )
        self.status_label.pack(side=tk.LEFT, padx=4)
        
        ttk.Label(self.status_bar, text="Image Count:").pack(side=tk.RIGHT, padx=4)
        self.count_label = ttk.Label(self.status_bar, text="0", width=6)
        self.count_label.pack(side=tk.RIGHT)

    def _build_navigation(self):
        """Build navigation button group"""
        nav_frame = ttk.LabelFrame(self.control_panel, text="Data Operations")
        nav_frame.pack(pady=10, padx=5, fill=tk.X)


        buttons = [
            ("Select Folder", self.load_folder, "📂"),
            ("Create Dataset", self.create_dataset, "⚙️"),
            ("Show Data", self.show_dataset, "👁️")
        ]

        for text, cmd, icon in buttons:
            btn = ttk.Button(
                nav_frame,
                text=f" {icon} {text}",
                command=cmd,
                style='Bold.TButton',  # 使用自定义样式
            )
            btn.pack(pady=3, fill=tk.X)


    def _add_cluster_controls(self):
        """Optimize cluster control components"""
        cluster_frame = ttk.LabelFrame(self.control_panel, text="Cluster Settings")
        cluster_frame.pack(pady=10, padx=5, fill=tk.X)

        # Add model selection button
        ttk.Button(
            cluster_frame,
            text="Select Heatmap Model",
            command=self._select_heatmap_model,
            style='Bold.TButton'
        ).pack(pady=5, fill=tk.X)

        # Parameter input
        param_frame = ttk.Frame(cluster_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Number of Clusters:",style='Bold.TButton').pack(side=tk.LEFT)
        self.cluster_num = ttk.Spinbox(
            param_frame,
            from_=2, to=20,
            values=[2,3,5,8,10],
            width=8
        )
        self.cluster_num.pack(side=tk.RIGHT)
        self.cluster_num.set(10)

        # Execute button
        self.cluster_btn = ttk.Button(
            cluster_frame,
            text="▶ Start Clustering",
            command=self.run_clustering,
            style='Bold.TButton'
        )
        self.cluster_btn.pack(pady=5, fill=tk.X)

        # Progress bar
        self.progress = ttk.Progressbar(
            cluster_frame,
            orient=tk.HORIZONTAL,
            mode='determinate'
        )
        self.progress.pack(fill=tk.X, pady=5)


    def run_clustering(self):
        """执行聚类操作"""
        if not self.dataset:
            messagebox.showwarning("警告", "请先加载数据集！")
            return

        # 禁用按钮防止重复点击
        self.cluster_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0

        # 在后台线程执行聚类
        threading.Thread(
            target=self._perform_clustering,
            daemon=True
        ).start()

    def _perform_clustering(self):
        """实际执行聚类的方法"""

        # 初始化模型和配置
        cfg = OmegaConf.load("configs/train.yaml")
        self.cfg = cfg
        model = resnet34_Network_cluster(
            backbone=cfg.models.backbone,
            num_attributes=cfg.data.num_attributes,
            pretrained=cfg.models.pretrained
        ).to(cfg.device).float()

        # 创建聚类实例
        cluster_app = Cluster_App(
            Dataset=self.dataset,
            Data_loader=self.dataloader,
            model=model,
            cfg=cfg,
            n_clusters=int(self.cluster_num.get())
        )

        # 执行聚类并更新进度
        # def update_progress(p):
        #     self.progress["value"] = p*100
        #     self.root.update_idletasks()

        self.cluster_app = cluster_app  # 保存聚类实例
        # self.cluster_results = cluster_app.run_clustering(
        #     progress_callback=update_progress
        # )

        # 显示聚类结果
        self.root.after(0, self._show_cluster_results)



    def _show_cluster_results(self):
        """显示带评分的聚类结果"""
        self.clear_display()
        
        for cluster_id in range(int(self.cluster_num.get())):
            dataset_cluster, self.indices = self.cluster_app.giveAndPlot_kmeansImage(
                target_element=cluster_id
            )
            
            # 创建聚类分组容器
            cluster_frame = tk.LabelFrame(
                self.main_container,
                text=self._get_cluster_header(cluster_id, indices = self.indices),
                bg='#F0F0F0',
                font=('Arial', 10, 'bold')
            )
            cluster_frame.pack(pady=10, fill=tk.BOTH, expand=True)
            
            # 创建网格布局
            grid_frame = tk.Frame(cluster_frame)
            grid_frame.pack(padx=5, pady=5)
            
            # 显示图片和评分
            # idx = 0,1,2,3
            for idx in range(min(8, len(dataset_cluster))):  # 每类显示4张
                self._create_cluster_cell(grid_frame, dataset_cluster, idx, cluster_id)
        
    def _get_cluster_header(self, cluster_id, indices):
        """生成带统计信息的标题（增加空数据保护）"""
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        # 使用明确的长度检查代替真值判断
        has_scores = bool(self.scores)
        has_indices = len(indices) > 0
        
        base_title = f"类别 {cluster_id}"
        if has_indices:
            base_title += f" (共{len(indices)}张)"
        
        if not has_scores or not has_indices:
            return base_title
        
        # 过滤有效索引（处理numpy索引）
        valid_indices = [i for i in indices if i < len(self.scores)]
        if not valid_indices:
            return base_title + " | 无有效评分"
        
        # 获取当前簇的所有评分
        cluster_scores = [self.scores[i] for i in valid_indices]
        
        try:
            # 计算统计指标（增加异常捕获）
            metric = self.metric_var.get()
            scores = [s['Aesth_score'] for s in cluster_scores if 'Aesth_score' in s]
            
            if not scores:
                return base_title + " | 缺少评分字段"
                
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            
        except Exception as e:
            print(f"统计计算错误: {str(e)}")
            return base_title + " | 统计错误"

        stats_text = {
            'mean': f"平均分: {avg_score:.2f}",
            'max': f"最高分: {max_score:.2f}",
            'min': f"最低分: {min_score:.2f}"
        }
        
        return f"{base_title} | {stats_text.get(metric, '')}"

    def _create_cluster_cell(self, parent, dataset, idx, cluster_id):
        """创建带评分的聚类单元格"""
        cell = tk.Frame(
            parent,
            width=220,
            height=240,
            bg='white',
            relief='groove',
            borderwidth=2
        )
        cell.grid(row=idx//4, column=idx%4, padx=5, pady=5)
        

        # 显示图像
        

        if self.scores:
            pass
        else:
            tensor = dataset[idx]
            img = self.tensor_to_pil(tensor)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(cell, image=photo)
            img_label.image = photo
            img_label.pack(pady=2)
        

        
        # 显示评分信息
        # 目前来看评分数据和图片不能对应
        if self.scores:
            # 转换为稳定的列表结构
            index_list = np.array(self.indices).flatten().tolist()


            # 按Aesth评分降序排序
            sorted_index_list = sorted(
                index_list,
                key=lambda x: self.scores[x]['Aesth_score'],
                reverse=True
            )



            self.clusterID_get_indices[cluster_id] = sorted_index_list
            # print('')
            # print('self.clusterID_get_indices[cluster_id]：',self.clusterID_get_indices[cluster_id])

            # # 验证排序结果示例
            # print("排序前首元素:", index_list[0], "评分:", self.scores[index_list[0]]['Aesth_score'])
            # print("排序后首元素:", sorted_index_list[0], "评分:", self.scores[sorted_index_list[0]]['Aesth_score'])


            
            # 安全获取全局索引
            global_idx = sorted_index_list[idx]


            
            # 显示全局索引（新增部分）
            index_label = tk.Label(cell, 
                                text=f"Dataset Index: {global_idx}",
                                font=('Courier', 9, 'bold'),
                                bg='#F0F0F0')
            index_label.pack(side=tk.TOP, fill=tk.X)

            # 2. 再添加图片
            tensor = self.dataset[global_idx]
            img = self.tensor_to_pil(tensor)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(cell, image=photo)
            img_label.image = photo
            img_label.pack(pady=2)


            # 获取评分数据
            score_info = self.scores[global_idx]
            
            # 主评分
            score_text = f"Aesth: {score_info['Aesth_score']:.2f}"
            tk.Label(cell,
                    text=score_text,
                    bg='#FFD700',
                    font=('Arial', 9, 'bold')).pack(fill=tk.X)
            
            # 关键属性
            attr_frame = tk.Frame(cell, bg='white')
            attr_frame.pack()
            
            # for attr in ['color_harmony', 'rule_of_thirds']:
            #     tk.Label(attr_frame,
            #             text=f"{attr[:4]}:{score_info[attr]:.2f}",
            #             font=('Arial', 8),
            #             bg='white').pack(side=tk.LEFT, padx=2)
    

        # 在显示图片的部分添加点击事件绑定
        img_label.bind("<Button-1>", lambda e, cid=cluster_id: self._on_cluster_click(cid))
        return cell
    
    def _on_cluster_click(self, cluster_id):
        """处理聚类点击事件"""
        self.current_cluster_id = cluster_id
        self.notebook.select(self.aesth_tab)  # 切换到固定分析标签页
        self._update_analysis_content()

    def _update_analysis_content(self):
        """更新分析标签页内容（使用固定容器）"""
        # 清空旧内容
        for widget in self.analysis_container.winfo_children():
            widget.destroy()

        # 添加标题
        ttk.Label(self.analysis_container, 
                text=f"Cluster {self.current_cluster_id}",
                font=('Arial', 14, 'bold')).pack(pady=10)

        # 获取并排序索引
        indices = self.clusterID_get_indices[self.current_cluster_id]
        sorted_indices = indices[:8]
        # 显示图片网格
        grid_frame = ttk.Frame(self.analysis_container)
        grid_frame.pack(pady=10)
        
        for idx, global_idx in enumerate(sorted_indices):
            # print("attr-----------------")
            # print("sorted_indices:", sorted_indices)
            # print("global_idx:", global_idx)
            # print("idx:", idx)
            # print("sorted_indices[idx]:", sorted_indices[idx])

            self._create_analysis_image(grid_frame, global_idx, idx)
            
    def _create_analysis_image(self, parent, global_idx, position):
        """Create an analysis page image unit with heatmap"""
        # Create the main container
        cell = ttk.Frame(parent)
        cell.grid(row=position, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create a Notebook for switchable content (increased size)
        img_notebook = ttk.Notebook(cell, width=900, height=580)  # Increased size
        img_notebook.pack(fill=tk.BOTH, expand=True)

        # Original image tab
        orig_frame = ttk.Frame(img_notebook)
        self._create_base_image(orig_frame, global_idx)
        img_notebook.add(orig_frame, text="Original Image")

        # Heatmap tab (3x5 layout)
        heatmap_frame = ttk.Frame(img_notebook)
        self._create_heatmap_content(heatmap_frame, global_idx)  # Modified layout method
        img_notebook.add(heatmap_frame, text="Heatmap Analysis")

        return cell

    def _create_base_image(self, parent, global_idx):
        """Create the base image display and integrate the analysis interface"""
        # Main container: horizontally arrange the image and analysis interface
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ================== Left-side image display area ===========# ...Omitted other code...


        # ==== Modified code ====
        img_frame = ttk.Frame(main_frame, width=300)
        img_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Display image
        tensor = self.dataset[global_idx]
        img = self.tensor_to_pil(tensor)
        img.thumbnail((280, 280))  # Slightly increase thumbnail size
        photo = ImageTk.PhotoImage(img)
        
        lbl = ttk.Label(img_frame, image=photo)
        lbl.image = photo
        lbl.pack(pady=5)
        
        # Bind image click event
        lbl.bind("<Button-1>", lambda e: self._trigger_analysis(global_idx))
        
        # ================== Right-side analysis interface area ===========# ...Omitted other code...


        # ==== Modified code ====
        analysis_frame = ttk.LabelFrame(
            main_frame,
            text="Aesthetic Analysis Results",
            padding=(10, 5),
            style='Analysis.TLabelframe'
        )
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        # Create independent analysis textbox (no longer use self.analysis_text)
        analysis_text = tk.Text(
            analysis_frame,
            wrap=tk.WORD,
            height=15,
            font=('Microsoft YaHei', 11),
            bg='#F8F9FA',
            relief=tk.FLAT
        )

        # Store in dictionary
        self.analysis_texts[global_idx] = analysis_text
        
        # Configure scrollbar
        scrollbar = ttk.Scrollbar(analysis_frame, command=analysis_text.yview)
        analysis_text.configure(yscrollcommand=scrollbar.set)
        
        # Layout
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Analysis button
        btn_frame = ttk.Frame(img_frame, padding=5)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        # Modify button callback
        ttk.Button(
            btn_frame,
            text="⭐ Start Analysis",
            command=lambda: self._trigger_analysis(global_idx),
            style='Accent.TButton'
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            btn_frame,
            text="🗑️ Clear Results",
            command=lambda: analysis_text.delete(1.0, tk.END),
            style='Secondary.TButton'
        ).pack(side=tk.RIGHT)

    def _trigger_analysis(self, global_idx):
        """Trigger analysis for the specified index image"""
        # Update the current active index
        self.current_active_idx = global_idx
        
        # Get the corresponding textbox
        text_widget = self.analysis_texts.get(global_idx)
        if not text_widget:
            return
        
        # Clear old content
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, "Analyzing, please wait...")
        text_widget.update()
        
        # Start asynchronous analysis
        threading.Thread(
            target=self._async_analyze_image,
            args=(global_idx,),
            daemon=True
        ).start()


    def _create_heatmap_content(self, parent, global_idx):
        """创建3x5网格布局的热力图"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动区域
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        container = ttk.Frame(canvas)
        canvas.create_window((0,0), window=container, anchor="nw")
        
        # 生成热力图内容
        if global_idx not in self.heatmap_cache:
            self._generate_heatmaps(global_idx)
        
        # 3x5网格布局
        row, col = 0, 0
        for idx, heatmap in enumerate(self.heatmap_cache[global_idx]):
            frame = ttk.Frame(container)
            frame.grid(row=row, column=col, padx=5, pady=5)
            
            # 显示单个热力图
            self._show_single_heatmap(frame, idx, heatmap, global_idx)
            
            # 更新网格位置
            col += 1
            if col >= 5:  # 每行5列
                col = 0
                row += 1
        
        # 配置画布滚动区域
        container.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # 绑定鼠标滚轮滚动
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

    def generate_contour_overlay(heatmap, 
                                 original_image, 
                                 threshold=0.15, 
                                 contour_color=(0, 255, 0), 
                                 thickness=2):
        """
        根据Grad-CAM热力图的形状绘制轮廓并叠加在原始图像上。
        
        参数：
            heatmap (numpy.ndarray): Grad-CAM热力图，通常是二维数组。
            original_image (numpy.ndarray): 原始输入图像。
            threshold (float): 二值化的阈值，范围在[0, 1]之间，默认为0.15。
            contour_color (tuple): 轮廓颜色，BGR格式，默认为绿色。
            thickness (int): 轮廓线条的厚度，默认为2。
        
        返回：
            numpy.ndarray: 绘制了轮廓的原始图像。
        """
        # 确保热力图是二维的
        if heatmap.ndim != 2:
            raise ValueError("热力图应该是二维的")
        
        # 归一化热力图到0-1范围
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # 应用阈值，生成二值图
        binary_map = heatmap > threshold
        
        # 查找连通区域
        contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在原始图像上绘制轮廓
        if contours:
            cv2.drawContours(original_image, contours, -1, contour_color, thickness)
        
        return original_image
    
    def _generate_heatmaps(self, global_idx):
        """生成并缓存热力图"""
        # 加载模型
        if not self._load_heatmap_model():
            return

        # 获取图像数据
        img_tensor = self.dataset[global_idx].unsqueeze(0).to(self.cfg.device).float()
        
        # 生成热力图
        Image_List = []
        cam_map_List = []
        
        # 原始图像
        img_np = np.array(self.tensor_to_pil(img_tensor.squeeze(0)))
        Image_List.append(img_np)

        # 各属性热力图
        for k in range(12):
            image_result, cam_map = show_heatmap_12dim(
                img_tensor=img_tensor,
                class_id=k,
                dataset=self.dataset,
                model_Dev=self.heatmap_model,
                global_idx=global_idx,
            )
            Image_List.append(image_result)
            cam_map_List.append(cam_map)

        # 综合热力图
        sum_cam_map = np.zeros_like(cam_map_List[0])
        for cam in cam_map_List[1:]:
            sum_cam_map += cam
        sum_cam_map = (sum_cam_map - sum_cam_map.min()) / (sum_cam_map.max() - sum_cam_map.min())
        # 修正为
        img_path = self.dataset.image_paths[global_idx]  # 使用正确的属性名
        img_pil = Image.open(img_path)
        sum_cam_map_image = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.6) # alpha越小，原图越淡
        Image_List.append(sum_cam_map_image)

        # 生成带有轮廓的图像
        image_with_contour = ImageDatasetApp.generate_contour_overlay(
            cam_map_List[3], 
            img_np.copy(),  # 使用原始图像的副本
            threshold=0.5,  # 调整阈值
            contour_color=(255, 0, 0),  # 红色轮廓
            thickness=2  # 轮廓线厚度
        )
        # print("生成轮廓: ",image_with_contour.shape)

        # 将生成的图像添加到 Image_List 中
        Image_List.append(image_with_contour)

        # 缓存结果
        self.heatmap_cache[global_idx] = Image_List
        # print(f"生成热力图: {global_idx}")

    


    def _show_single_heatmap(self, parent, idx, heatmap, global_idx):
        """显示单个热力图单元（调整为适合网格布局）"""
        frame = ttk.Frame(parent)
        

        # 调整显示尺寸
        display_size = 150  # 缩小显示尺寸以适应网格
        if isinstance(heatmap, np.ndarray):
            img = Image.fromarray(heatmap).resize((display_size, display_size))
        else:
            img = heatmap.resize((display_size, display_size))
        
        photo = ImageTk.PhotoImage(img)
        
        # 属性名称列表
        attr_names = [
            'Origin image',                # 0
            'Aesth_score',        # 1 -> Aesth_score
            'balancing_ele',     # 2 -> balancing_ele
            'color_harmony',          # 3 -> color_harmony 
            'content',                # 4 -> content
            'depth_of_field',         # 5 -> depth_of_field
            'light',          # 6 -> light
            'motion_blur',            # 7 -> motion_blur
            'object',           # 8 -> object
            'repetition',             # 9 -> repetition
            'rule_of_thirds',         # 10 -> rule_of_thirds
            'symmetry',               # 11 -> symmetry
            'vivid_color',            # 12 -> vivid_color
            'KAN_scores',                 # 13 (需要确认是否存在对应键)
            'box'
        ]
        
        
        # 获取评分文本
        score_text = ""
        if idx > 0:  # 跳过原始图像
            try:
                scores = self.scores[global_idx]
                # print(f"获取评分: scores",scores)
                if idx >= len(attr_names)-2:  # 综合评分
                    score_text = ''
                else:
                    # # 将属性名转换为小写作为键（如Aesth -> aesth）
                    key = attr_names[idx]
                    score_text = f"\nscores: {scores[key]:.2f}"
            except (IndexError, KeyError) as e:
                score_text = "\nscores: N/A"
                print(f"评分获取错误: {str(e)}")

        # 创建带评分的标签
        label = ttk.Label(
            frame, 
            image=photo, 
            text=f"{attr_names[idx]}{score_text}",
            compound="top", 
            padding=2,
            font=('Microsoft YaHei', 9),  # 使用更清晰的中文字体
            foreground='#333333'         # 深灰色文字
        )
        label.image = photo
        label.pack()
            

        
        frame.pack()

    def _load_heatmap_model(self):
            """加载热力图生成模型"""
            try:
                if self.heatmap_model is None:
                    # 如果未选择模型路径，则使用默认路径
                    if not hasattr(self, 'heatmap_model_path'):
                        self.heatmap_model_path = '/home/zl/下载/input/pykan-master/models/Data_argu/Data_argu_4_depth_of_field_epoch_8_loss_0.0548_grid_1_score_0.41242475441672044.pt'

                    # 初始化模型
                    self.heatmap_model = RegressionNetwork_kan(
                        backbone=cfg.models.backbone,
                        num_attributes=12,
                        pretrained=cfg.models.pretrained,
                        kan=None,
                    ).to(self.cfg.device).eval()

                    # 加载模型权重
                    self.heatmap_model.load_state_dict(torch.load(self.heatmap_model_path))
                return True
            except Exception as e:
                messagebox.showerror("错误", f"加载热力图模型失败：{str(e)}")
                return False

    def _select_heatmap_model(self):
        """选择热力图模型文件"""
        initial_dir = '/home/zl/下载/input/pykan-master/models'  # 默认模型目录
        filetypes = [('PyTorch模型', '*.pt'), ('所有文件', '*.*')]
        
        filepath = filedialog.askopenfilename(
            title="选择热力图模型文件",
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        if filepath:
            if not filepath.endswith('.pt'):
                messagebox.showwarning("警告", "请选择有效的PyTorch模型文件(.pt)")
                return
            
            # 更新模型路径
            self.heatmap_model_path = filepath
            self.heatmap_model = None  # 重置模型实例
            messagebox.showinfo("提示", "模型路径已更新，下次生成热力图时将使用新模型")


    def _show_score_info(self, parent, global_idx):
        """显示评分信息"""
        score = self.scores[global_idx]
        info = "\n".join([
            f"Aesth: {score['Aesth_score']:.2f}",
            f"Color: {score['color_harmony']:.2f}",
            f"Composition: {score['rule_of_thirds']:.2f}"
        ])
        
        ttk.Label(parent,
                text=info,
                font=('Courier', 8),
                relief="groove").pack(fill=tk.X, pady=2)

        
    
    def _show_score_info(self, parent, global_idx):
        """显示评分信息"""
        # 获取评分数据
        score_info = self.scores[global_idx]

    def _update_analysis_tab(self):
        """更新分析标签页内容"""
        # 清空旧内容
        for widget in self.cluster_tab.winfo_children():
            widget.destroy()

        # 创建新的滚动容器
        container = self.setup_image_grid(self.cluster_tab)
        
        # 添加标题
        ttk.Label(container, 
                text=f"Cluster {self.current_cluster_id} 详细分析",
                font=('Arial', 14, 'bold')).pack(pady=10)

        # 获取当前聚类的索引
        indices = self.clusterID_get_indices[self.current_cluster_id]
        
        # 生成排序后的索引列表
        self.sorted_index_list = sorted(
            indices,
            key=lambda x: self.scores[x]['Aesth_score'],
            reverse=True
        )[:8]

        # 显示排序后的图片
        grid_frame = ttk.Frame(container)
        grid_frame.pack(pady=10)
        
        for idx, global_idx in enumerate(self.sorted_index_list):
            self._create_analysis_image(grid_frame, global_idx, idx)


    # 修改原有显示方法
    def show_dataset(self):
        """显示数据集（根据是否聚类显示不同视图）"""

        print("show_dataset")
        # 修改判断条件为同时检查聚类结果和评分数据
        if self.cluster_app is not None:
            # 显示聚类结果
            self.root.after(0, self._show_cluster_results)

            # self._show_cluster_results()
        else:
            self._show_raw_dataset()

    def _show_raw_dataset(self):
        """显示原始数据集"""
        """显示数据集中的图片"""
        if not self.dataset:
            messagebox.showwarning("警告", "请先创建数据集！")
            return

        self.clear_display()
        self.create_image_grid(self.dataset)


    def create_widgets(self):
        """创建控制面板"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10, fill=tk.X)

        # 路径显示
        self.path_label = tk.Label(control_frame, text="当前路径：未选择", width=60, anchor='w')
        self.path_label.pack(side=tk.LEFT, padx=5)

        # 功能按钮
        tk.Button(
            control_frame,
            text="选择图片文件夹",
            command=self.load_folder
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="创建数据集",
            command=self.create_dataset
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="显示数据集",
            command=self.show_dataset
        ).pack(side=tk.LEFT, padx=5)

    def setup_image_grid(self, parent):
        """图片显示区域优化"""
        # 滚动系统
        canvas = tk.Canvas(parent, bg=self.colors['background'])
        scroll_y = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_x = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=canvas.xview)
        
        canvas.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )
        
        # 布局
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 图片容器
        self.main_container = ttk.Frame(canvas)
        canvas.create_window((0,0), window=self.main_container, anchor=tk.NW)

        # 事件绑定
        self.main_container.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    def _create_image_cell(self, parent, dataset, idx):
        """创建单个图片单元格"""
        cell = tk.Frame(parent, width=150, height=170, 
                       borderwidth=1, relief='groove')
        
        try:
            # 显示图像
            tensor = dataset[idx]
            img = self.tensor_to_pil(tensor)
            img.thumbnail((140, 140))
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(cell, image=photo)
            img_label.image = photo
            img_label.pack()
            
            # 显示评分
            if idx < len(self.scores):
                score = self.scores[idx]['Aesth_score']
                tk.Label(cell, 
                        text=f"Score: {score:.2f}",
                        bg='#FFD700', fg='black',
                        font=('Arial', 9)).pack(fill=tk.X)
                
        except Exception as e:
            tk.Label(cell, text=f"错误\n{str(e)[:15]}", fg='red').pack()
        
        return cell


    def load_folder(self):
        """Load Image Folder"""
        path = filedialog.askdirectory()
        if path:
            self.current_path = path
            self.path_label.config(text=f"Current Path: {path[:50]}...")
            messagebox.showinfo("Info", f"Successfully loaded path: {path}")

    def create_dataset(self):
        """Create Dataset and Data Loader"""
        if not self.current_path:
            messagebox.showwarning("Warning", "Please select an image folder first!")
            return

        try:
            self.dataset, self.dataloader = self.create_data_loader(self.current_path)
            messagebox.showinfo("Success", 
                f"Dataset creation completed!\n"
                f"Number of samples: {len(self.dataset)}\n"
                f"Batch shape: {next(iter(self.dataloader)).shape}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset: {str(e)}")

        # def show_dataset(self):
        #     """显示数据集中的图片"""
        #     if not self.dataset:
        #         messagebox.showwarning("警告", "请先创建数据集！")
        #         return

        #     self.clear_display()
        #     self.create_image_grid(self.dataset)

    # 修改数据集显示方法
    def create_image_grid(self, dataset):
        """创建带评分的图片网格"""
        container = self.main_container
        cols = 4  # 减少每行列数以容纳更多信息
        
        for idx in range(len(dataset)):
            row = idx // cols
            col = idx % cols
            
            if col == 0:
                row_frame = tk.Frame(container)
                row_frame.pack(pady=5, anchor='w')

            cell = self._create_image_cell(row_frame, dataset, idx)
            cell.grid(row=0, column=col, padx=5)

    def tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        # 处理不同形状的张量
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.size(0) > 3:
            tensor = tensor[:3]
        
        # 反标准化
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        return T.ToPILImage()(tensor)

    def clear_display(self):
        """清空显示区域"""
        for widget in self.main_container.winfo_children():
            widget.destroy()

    def on_mousewheel(self, event):
        """处理鼠标滚轮滚动"""
        self.canvas.yview_scroll(-1*(event.delta//120), "units")

    @staticmethod
    def create_data_loader(path):
        """创建数据加载器（从用户代码迁移）"""
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

        class CustomDataset(Dataset):
            def __init__(self, root_dir, transform=None, max_samples=1000):
                self.root_dir = root_dir
                self.transform = transform
                self.image_paths = self._load_paths(max_samples)

            def _load_paths(self, max_samples):
                valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
                return [
                    os.path.join(self.root_dir, f) 
                    for f in os.listdir(self.root_dir)[:max_samples] 
                    if f.lower().endswith(valid_ext)
                ]

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                try:
                    img = Image.open(self.image_paths[idx]).convert('RGB')
                    return self.transform(img) if self.transform else img
                except:
                    return torch.zeros(3, 256, 256)

        dataset = CustomDataset(path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        return dataset, dataloader

