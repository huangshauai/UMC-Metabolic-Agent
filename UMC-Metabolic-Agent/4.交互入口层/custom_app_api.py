# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 自定义应用API模块（RESTful API+异步任务+标准化接口）
核心逻辑：基于FastAPI构建HTTP接口，封装智能体全功能，支持外部应用集成/二次开发
设计原则：标准化、异步化、易集成、高可用，适配开发者快速对接UMC智能体能力
"""
import uvicorn
import fastapi
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union, Literal
import json
import os
import time
import uuid
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum

# 导入核心模块（兼容未安装/未找到情况）
try:
    from universal_cmd import UniversalCmd
    from tuner_dashboard import TunerDashboard
    from result_analysis import ResultAnalyzer
    from report_generator import ReportGenerator
    CORE_MODULES_LOADED = True
except ImportError as e:
    warnings.warn(f"核心模块导入失败：{e}\n部分API功能将受限，建议确保核心文件在当前目录")
    CORE_MODULES_LOADED = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("umc_api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UMC-API")

warnings.filterwarnings("ignore")

# ------------------------------ 常量定义 ------------------------------
# API版本
API_VERSION = "v1"
# 基础目录
BASE_DIR = os.getcwd()
# 输出目录
OUTPUT_DIR = "./umc_api_output"
# 任务队列目录
TASKS_DIR = "./umc_api_tasks"
# 上传文件目录
UPLOAD_DIR = "./umc_api_uploads"

# 创建目录
for dir_path in [OUTPUT_DIR, TASKS_DIR, UPLOAD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ------------------------------ 数据模型 ------------------------------
class DomainEnum(str, Enum):
    """领域枚举"""
    GENERAL = "general"
    QUANTUM = "quantum"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    FINANCE = "finance"

class TaskStatusEnum(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RunRequest(BaseModel):
    """智能体运行请求模型"""
    data_path: str = Field(..., description="输入数据文件路径（相对/绝对）")
    domain: DomainEnum = Field(DomainEnum.GENERAL, description="目标领域")
    run_time: int = Field(300, ge=10, le=3600, description="运行时长（秒）")
    output_name: str = Field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}", description="输出文件名")

    @validator("data_path")
    def validate_data_path(cls, v):
        """验证数据文件存在"""
        if not os.path.exists(v) and not v.startswith("/uploads/"):
            raise ValueError(f"数据文件不存在：{v}")
        # 处理上传文件路径
        if v.startswith("/uploads/"):
            real_path = os.path.join(UPLOAD_DIR, v.replace("/uploads/", ""))
            if not os.path.exists(real_path):
                raise ValueError(f"上传文件不存在：{real_path}")
            return real_path
        return v

class TunerRequest(BaseModel):
    """调优请求模型"""
    data_path: str = Field(..., description="调优数据文件路径")
    domain: DomainEnum = Field(DomainEnum.GENERAL, description="目标领域")
    adapt_iterations: int = Field(50, ge=10, le=500, description="调优迭代次数")
    learning_rate: float = Field(0.01, ge=0.001, le=0.1, description="学习率")
    core_factor_weight: float = Field(0.8, ge=0.1, le=1.0, description="核心因子权重")
    target_metric: str = Field("metabolic_efficiency", description="优化目标指标")

    @validator("data_path")
    def validate_data_path(cls, v):
        """验证数据文件存在"""
        if not os.path.exists(v) and not v.startswith("/uploads/"):
            raise ValueError(f"数据文件不存在：{v}")
        if v.startswith("/uploads/"):
            real_path = os.path.join(UPLOAD_DIR, v.replace("/uploads/", ""))
            if not os.path.exists(real_path):
                raise ValueError(f"上传文件不存在：{real_path}")
            return real_path
        return v

class AnalyzeRequest(BaseModel):
    """分析请求模型"""
    data_path: str = Field(..., description="分析数据文件路径")
    target_col: str = Field("metabolic_efficiency", description="目标分析列名")
    analysis_types: List[str] = Field(["basic", "feature"], description="分析类型")

class ReportRequest(BaseModel):
    """报告生成请求模型"""
    analysis_path: str = Field(..., description="分析结果路径（文件/目录）")
    report_type: str = Field("comprehensive", description="报告类型")
    formats: List[str] = Field(["md", "html"], description="报告格式")
    with_plots: bool = Field(True, description="是否包含图表")

class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatusEnum = Field(..., description="任务状态")
    message: str = Field(..., description="任务消息")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果（完成后）")
    error: Optional[str] = Field(None, description="错误信息（失败时）")

# ------------------------------ API核心类 ------------------------------
class UMCCustomAPI:
    """UMC智能体自定义API控制器"""
    def __init__(self):
        """初始化API控制器"""
        # 创建FastAPI应用
        self.app = FastAPI(
            title="UMC-Metabolic-Agent 自定义API",
            description="UMC智能体RESTful API接口，支持运行/调优/分析/报告全功能",
            version=API_VERSION,
            docs_url="/docs",  # Swagger文档
            redoc_url="/redoc"  # ReDoc文档
        )
        
        # 配置跨域
        self._setup_cors()
        
        # 初始化核心模块
        self.cmd = UniversalCmd() if CORE_MODULES_LOADED else None
        self.tuner = TunerDashboard() if CORE_MODULES_LOADED else None
        self.analyzer = ResultAnalyzer(output_dir=f"{OUTPUT_DIR}/analysis") if CORE_MODULES_LOADED else None
        self.report_generator = ReportGenerator(output_dir=f"{OUTPUT_DIR}/reports") if CORE_MODULES_LOADED else None
        
        # 任务存储（内存+文件持久化）
        self.tasks: Dict[str, TaskResponse] = {}
        self._load_tasks()
        
        # 注册路由
        self._register_routes()

    def _setup_cors(self):
        """配置跨域中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境请指定具体域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self):
        """注册API路由"""
        # 健康检查
        @self.app.get("/health", tags=["基础接口"], summary="健康检查")
        async def health_check():
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "api_version": API_VERSION,
                    "core_modules_loaded": CORE_MODULES_LOADED,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # 文件上传
        @self.app.post("/upload", tags=["文件操作"], summary="上传数据文件")
        async def upload_file(file: UploadFile = File(..., description="CSV/Excel数据文件")):
            try:
                # 生成唯一文件名
                file_ext = file.filename.split(".")[-1]
                file_name = f"{uuid.uuid4()}.{file_ext}"
                file_path = os.path.join(UPLOAD_DIR, file_name)
                
                # 保存文件
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                
                logger.info(f"文件上传成功：{file_name}")
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "file_name": file_name,
                        "file_path": f"/uploads/{file_name}",
                        "size": os.path.getsize(file_path),
                        "upload_time": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"文件上传失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"文件上传失败：{str(e)}")
        
        # 获取上传文件列表
        @self.app.get("/uploads", tags=["文件操作"], summary="获取上传文件列表")
        async def list_uploads():
            try:
                files = []
                for file in os.listdir(UPLOAD_DIR):
                    if file.endswith((".csv", ".xlsx", ".xls")):
                        file_path = os.path.join(UPLOAD_DIR, file)
                        files.append({
                            "file_name": file,
                            "file_path": f"/uploads/{file}",
                            "size": os.path.getsize(file_path),
                            "create_time": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                        })
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "count": len(files),
                        "files": files
                    }
                )
            except Exception as e:
                logger.error(f"获取上传文件列表失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"获取文件列表失败：{str(e)}")
        
        # 智能体运行（同步）
        @self.app.post("/run", tags=["核心功能"], summary="运行UMC智能体（同步）")
        async def run_agent(request: RunRequest):
            try:
                if not self.cmd:
                    raise HTTPException(status_code=503, detail="核心模块未加载，无法执行运行操作")
                
                # 执行运行
                run_args = fastapi.datastructures.FormData(
                    data_path=request.data_path,
                    domain=request.domain.value,
                    run_time=request.run_time,
                    output_path=f"{OUTPUT_DIR}/run/{request.output_name}.csv"
                )
                
                result = self.cmd._execute_run(run_args, return_result=True)
                
                logger.info(f"智能体运行完成：{request.output_name}")
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "task_type": "run",
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"智能体运行失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"运行失败：{str(e)}")
        
        # 智能体运行（异步）
        @self.app.post("/run/async", tags=["核心功能"], summary="运行UMC智能体（异步）")
        async def run_agent_async(request: RunRequest, background_tasks: BackgroundTasks):
            try:
                # 创建任务
                task_id = str(uuid.uuid4())
                task = TaskResponse(
                    task_id=task_id,
                    status=TaskStatusEnum.PENDING,
                    message="任务已创建，等待执行"
                )
                self.tasks[task_id] = task
                self._save_task(task)
                
                # 定义后台任务
                def run_background_task(task_id: str, request: RunRequest):
                    try:
                        # 更新任务状态
                        self.tasks[task_id].status = TaskStatusEnum.RUNNING
                        self.tasks[task_id].message = "任务正在执行"
                        self._save_task(self.tasks[task_id])
                        
                        if not self.cmd:
                            raise Exception("核心模块未加载，无法执行运行操作")
                        
                        # 执行运行
                        run_args = fastapi.datastructures.FormData(
                            data_path=request.data_path,
                            domain=request.domain.value,
                            run_time=request.run_time,
                            output_path=f"{OUTPUT_DIR}/run/{request.output_name}.csv"
                        )
                        
                        result = self.cmd._execute_run(run_args, return_result=True)
                        
                        # 更新任务结果
                        self.tasks[task_id].status = TaskStatusEnum.COMPLETED
                        self.tasks[task_id].message = "任务执行完成"
                        self.tasks[task_id].result = result
                        self._save_task(self.tasks[task_id])
                        
                        logger.info(f"异步任务完成：{task_id}")
                        
                    except Exception as e:
                        logger.error(f"异步任务失败：{task_id} - {str(e)}")
                        self.tasks[task_id].status = TaskStatusEnum.FAILED
                        self.tasks[task_id].message = "任务执行失败"
                        self.tasks[task_id].error = str(e)
                        self._save_task(self.tasks[task_id])
                
                # 添加后台任务
                background_tasks.add_task(run_background_task, task_id, request)
                
                return JSONResponse(
                    status_code=202,
                    content={
                        "success": True,
                        "task_id": task_id,
                        "status": "pending",
                        "message": "异步任务已创建",
                        "task_url": f"/tasks/{task_id}",
                        "docs_url": "/docs"
                    }
                )
            except Exception as e:
                logger.error(f"创建异步任务失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"创建异步任务失败：{str(e)}")
        
        # 智能体调优（异步）
        @self.app.post("/tuner", tags=["核心功能"], summary="智能体参数调优（异步）")
        async def tune_agent(request: TunerRequest, background_tasks: BackgroundTasks):
            try:
                # 创建任务
                task_id = str(uuid.uuid4())
                task = TaskResponse(
                    task_id=task_id,
                    status=TaskStatusEnum.PENDING,
                    message="调优任务已创建，等待执行"
                )
                self.tasks[task_id] = task
                self._save_task(task)
                
                # 定义后台调优任务
                def tune_background_task(task_id: str, request: TunerRequest):
                    try:
                        # 更新任务状态
                        self.tasks[task_id].status = TaskStatusEnum.RUNNING
                        self.tasks[task_id].message = "调优任务正在执行"
                        self._save_task(self.tasks[task_id])
                        
                        if not self.tuner:
                            raise Exception("调优模块未加载，无法执行调优操作")
                        
                        # 更新调优参数
                        self.tuner.default_params.update({
                            "domain": request.domain.value,
                            "adapt_iterations": request.adapt_iterations,
                            "learning_rate": request.learning_rate,
                            "core_factor_weight": request.core_factor_weight,
                            "target_metric": request.target_metric
                        })
                        
                        # 执行调优
                        self.tuner._start_tuner(request.data_path)
                        
                        # 等待调优完成（简化版，实际生产环境建议用任务队列）
                        while self.tuner.tuner_status["is_running"]:
                            time.sleep(1)
                        
                        # 构建调优结果
                        tune_result = {
                            "domain": request.domain.value,
                            "iterations": request.adapt_iterations,
                            "best_score": self.tuner.tuner_status["best_score"],
                            "best_params": self.tuner.tuner_status["best_params"],
                            "duration": self.tuner.tuner_status["elapsed_time"],
                            "output_dir": OUTPUT_DIR
                        }
                        
                        # 保存调优记录
                        tune_record = {
                            "task_id": task_id,
                            "timestamp": datetime.now().isoformat(),
                            **tune_result
                        }
                        with open(f"{TASKS_DIR}/{task_id}_tune_result.json", "w", encoding="utf-8") as f:
                            json.dump(tune_record, f, ensure_ascii=False, indent=2)
                        
                        # 更新任务结果
                        self.tasks[task_id].status = TaskStatusEnum.COMPLETED
                        self.tasks[task_id].message = "调优任务执行完成"
                        self.tasks[task_id].result = tune_result
                        self._save_task(self.tasks[task_id])
                        
                        logger.info(f"调优任务完成：{task_id}")
                        
                    except Exception as e:
                        logger.error(f"调优任务失败：{task_id} - {str(e)}")
                        self.tasks[task_id].status = TaskStatusEnum.FAILED
                        self.tasks[task_id].message = "调优任务执行失败"
                        self.tasks[task_id].error = str(e)
                        self._save_task(self.tasks[task_id])
                
                # 添加后台任务
                background_tasks.add_task(tune_background_task, task_id, request)
                
                return JSONResponse(
                    status_code=202,
                    content={
                        "success": True,
                        "task_id": task_id,
                        "status": "pending",
                        "message": "调优任务已创建",
                        "task_url": f"/tasks/{task_id}",
                        "estimated_time": f"{request.adapt_iterations * 0.1:.1f}秒"
                    }
                )
            except Exception as e:
                logger.error(f"创建调优任务失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"创建调优任务失败：{str(e)}")
        
        # 结果分析
        @self.app.post("/analyze", tags=["核心功能"], summary="结果分析")
        async def analyze_result(request: AnalyzeRequest):
            try:
                if not self.analyzer:
                    raise HTTPException(status_code=503, detail="分析模块未加载，无法执行分析操作")
                
                # 执行分析
                analyze_args = fastapi.datastructures.FormData(
                    data_path=request.data_path,
                    target_col=request.target_col,
                    analysis_types=request.analysis_types,
                    output_path=f"{OUTPUT_DIR}/analysis/{uuid.uuid4()}"
                )
                
                result = self.cmd._execute_analyze(analyze_args, return_result=True) if self.cmd else {}
                
                logger.info(f"分析完成：{request.data_path}")
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "analysis_types": request.analysis_types,
                        "target_col": request.target_col,
                        "result": result,
                        "output_dir": analyze_args.output_path,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"分析失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"分析失败：{str(e)}")
        
        # 生成报告
        @self.app.post("/report", tags=["核心功能"], summary="生成分析报告")
        async def generate_report(request: ReportRequest):
            try:
                if not self.report_generator:
                    raise HTTPException(status_code=503, detail="报告模块未加载，无法生成报告")
                
                # 执行报告生成
                report_args = fastapi.datastructures.FormData(
                    analysis_path=request.analysis_path,
                    report_type=request.report_type,
                    format=request.formats,
                    with_plots=request.with_plots,
                    output_name=f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                
                self.cmd._execute_report(report_args) if self.cmd else None
                
                # 构建报告路径
                report_paths = {}
                for fmt in request.formats:
                    report_path = f"{OUTPUT_DIR}/reports/{report_args.output_name}.{fmt}"
                    if os.path.exists(report_path):
                        report_paths[fmt] = report_path
                
                logger.info(f"报告生成完成：{report_args.output_name}")
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "report_type": request.report_type,
                        "formats": request.formats,
                        "report_paths": report_paths,
                        "with_plots": request.with_plots,
                        "download_urls": {fmt: f"/download/{os.path.basename(path)}" for fmt, path in report_paths.items()}
                    }
                )
            except Exception as e:
                logger.error(f"生成报告失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"生成报告失败：{str(e)}")
        
        # 获取任务状态
        @self.app.get("/tasks/{task_id}", tags=["任务管理"], summary="获取任务状态/结果")
        async def get_task(task_id: str):
            try:
                if task_id not in self.tasks:
                    # 尝试从文件加载
                    task_file = os.path.join(TASKS_DIR, f"{task_id}.json")
                    if os.path.exists(task_file):
                        with open(task_file, "r", encoding="utf-8") as f:
                            task_data = json.load(f)
                            self.tasks[task_id] = TaskResponse(**task_data)
                    else:
                        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
                
                task = self.tasks[task_id]
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "task": task.dict()
                    }
                )
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"获取任务失败：{task_id} - {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取任务失败：{str(e)}")
        
        # 获取任务列表
        @self.app.get("/tasks", tags=["任务管理"], summary="获取任务列表")
        async def list_tasks(
            status: Optional[TaskStatusEnum] = Query(None, description="任务状态筛选"),
            limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
            offset: int = Query(0, ge=0, description="偏移量")
        ):
            try:
                # 筛选任务
                task_list = list(self.tasks.values())
                if status:
                    task_list = [t for t in task_list if t.status == status]
                
                # 分页
                task_list = task_list[offset:offset+limit]
                
                # 转换为字典
                tasks_data = [t.dict() for t in task_list]
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "count": len(tasks_data),
                        "total": len(self.tasks),
                        "limit": limit,
                        "offset": offset,
                        "tasks": tasks_data
                    }
                )
            except Exception as e:
                logger.error(f"获取任务列表失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"获取任务列表失败：{str(e)}")
        
        # 文件下载
        @self.app.get("/download/{file_name}", tags=["文件操作"], summary="下载生成的文件")
        async def download_file(file_name: str):
            try:
                # 查找文件
                file_paths = [
                    os.path.join(OUTPUT_DIR, "run", file_name),
                    os.path.join(OUTPUT_DIR, "analysis", file_name),
                    os.path.join(OUTPUT_DIR, "reports", file_name),
                    os.path.join(UPLOAD_DIR, file_name),
                    os.path.join(TASKS_DIR, file_name)
                ]
                
                file_path = None
                for path in file_paths:
                    if os.path.exists(path):
                        file_path = path
                        break
                
                if not file_path:
                    raise HTTPException(status_code=404, detail=f"文件不存在：{file_name}")
                
                # 返回文件
                return FileResponse(
                    path=file_path,
                    filename=file_name,
                    media_type="application/octet-stream"
                )
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"文件下载失败：{file_name} - {str(e)}")
                raise HTTPException(status_code=500, detail=f"文件下载失败：{str(e)}")
        
        # 获取配置信息
        @self.app.get("/config", tags=["系统管理"], summary="获取API配置信息")
        async def get_config():
            try:
                config = {
                    "api_version": API_VERSION,
                    "base_dir": BASE_DIR,
                    "output_dir": OUTPUT_DIR,
                    "upload_dir": UPLOAD_DIR,
                    "tasks_dir": TASKS_DIR,
                    "core_modules_loaded": CORE_MODULES_LOADED,
                    "supported_domains": [d.value for d in DomainEnum],
                    "supported_task_status": [s.value for s in TaskStatusEnum],
                    "server_time": datetime.now().isoformat()
                }
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "config": config
                    }
                )
            except Exception as e:
                logger.error(f"获取配置失败：{str(e)}")
                raise HTTPException(status_code=500, detail=f"获取配置失败：{str(e)}")

    def _save_task(self, task: TaskResponse):
        """保存任务到文件"""
        task_path = os.path.join(TASKS_DIR, f"{task.task_id}.json")
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(task.dict(), f, ensure_ascii=False, indent=2)

    def _load_tasks(self):
        """从文件加载任务"""
        try:
            for file in os.listdir(TASKS_DIR):
                if file.endswith(".json") and not file.endswith("_tune_result.json"):
                    task_id = file.replace(".json", "")
                    task_path = os.path.join(TASKS_DIR, file)
                    with open(task_path, "r", encoding="utf-8") as f:
                        task_data = json.load(f)
                        self.tasks[task_id] = TaskResponse(**task_data)
            logger.info(f"加载历史任务：{len(self.tasks)}条")
        except Exception as e:
            logger.warning(f"加载历史任务失败：{str(e)}")

    def run_server(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """启动API服务器"""
        logger.info(f"启动UMC API服务器：http://{host}:{port}")
        logger.info(f"API文档地址：http://{host}:{port}/docs")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

# ------------------------------ 快捷使用函数 ------------------------------
def create_api_app() -> FastAPI:
    """创建API应用实例（供外部使用）"""
    api = UMCCustomAPI()
    return api.app

def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """启动API服务器（快捷函数）"""
    api = UMCCustomAPI()
    api.run_server(host=host, port=port, reload=reload)

# ------------------------------ 命令行入口 ------------------------------
if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="UMC智能体自定义API服务器")
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0", help="服务器监听地址")
    parser.add_argument("--port", "-P", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="自动重载（开发模式）")
    
    args = parser.parse_args()
    
    # 启动服务器
    run_api_server(host=args.host, port=args.port, reload=args.reload)