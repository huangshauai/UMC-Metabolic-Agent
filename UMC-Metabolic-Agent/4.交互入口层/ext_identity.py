# -*- coding: utf-8 -*-
"""
UMC-Metabolic-Agent 外部身份认证模块（JWT认证+RBAC权限+多端统一管控）
核心逻辑：提供标准化身份认证和权限管理能力，适配API/仪表盘/命令行多端访问控制
设计原则：安全性、易用性、可扩展性，支持JWT令牌、API密钥、角色权限多级管控
"""
import jwt
import bcrypt
import json
import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
from pathlib import Path

# Pydantic用于数据验证（兼容未安装情况）
try:
    from pydantic import BaseModel, Field, validator, EmailStr
    PYDANTIC_LOADED = True
except ImportError:
    warnings.warn("Pydantic未安装，部分数据验证功能将受限")
    PYDANTIC_LOADED = False

# FastAPI/Streamlit集成（兼容未安装情况）
try:
    from fastapi import Request, HTTPException, Depends
    from fastapi.security import OAuth2PasswordBearer, APIKeyHeader, APIKeyQuery
    import streamlit as st
    FASTAPI_STREAMLIT_LOADED = True
except ImportError:
    warnings.warn("FastAPI/Streamlit未安装，中间件/仪表盘集成功能将受限")
    FASTAPI_STREAMLIT_LOADED = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("umc_identity.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UMC-Identity")

warnings.filterwarnings("ignore")

# ------------------------------ 常量定义 ------------------------------
# 认证配置
IDENTITY_CONFIG = {
    "JWT_SECRET_KEY": os.getenv("UMC_JWT_SECRET", "umc-metabolic-agent-2026-secret-key"),  # 生产环境请更换
    "JWT_ALGORITHM": "HS256",
    "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 120,  # 访问令牌过期时间（分钟）
    "JWT_REFRESH_TOKEN_EXPIRE_DAYS": 7,      # 刷新令牌过期时间（天）
    "API_KEY_HEADER_NAME": "X-UMC-API-Key",  # API密钥请求头名称
    "API_KEY_QUERY_NAME": "api_key",         # API密钥URL参数名称
}

# 数据存储路径
DATA_DIR = "./umc_identity_data"
USERS_FILE = f"{DATA_DIR}/users.json"
API_KEYS_FILE = f"{DATA_DIR}/api_keys.json"
PERMISSIONS_FILE = f"{DATA_DIR}/permissions.json"

# 创建目录
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------ 枚举定义 ------------------------------
class RoleEnum(str, Enum):
    """角色枚举（RBAC）"""
    ADMIN = "admin"       # 管理员：全部权限
    OPERATOR = "operator" # 操作员：运行/调优/分析权限
    VIEWER = "viewer"     # 查看者：仅查看/下载权限
    GUEST = "guest"       # 访客：仅基础访问权限

class PermissionEnum(str, Enum):
    """权限枚举"""
    # 核心功能权限
    AGENT_RUN = "agent:run"
    AGENT_TUNE = "agent:tune"
    AGENT_ANALYZE = "agent:analyze"
    AGENT_REPORT = "agent:report"
    # 管理权限
    USER_MANAGE = "user:manage"
    API_KEY_MANAGE = "api_key:manage"
    CONFIG_MANAGE = "config:manage"
    # 查看权限
    DATA_VIEW = "data:view"
    REPORT_VIEW = "report:view"
    TASK_VIEW = "task:view"

# 角色-权限映射（RBAC核心）
ROLE_PERMISSIONS = {
    RoleEnum.ADMIN: [p.value for p in PermissionEnum],
    RoleEnum.OPERATOR: [
        PermissionEnum.AGENT_RUN.value,
        PermissionEnum.AGENT_TUNE.value,
        PermissionEnum.AGENT_ANALYZE.value,
        PermissionEnum.AGENT_REPORT.value,
        PermissionEnum.DATA_VIEW.value,
        PermissionEnum.REPORT_VIEW.value,
        PermissionEnum.TASK_VIEW.value
    ],
    RoleEnum.VIEWER: [
        PermissionEnum.DATA_VIEW.value,
        PermissionEnum.REPORT_VIEW.value,
        PermissionEnum.TASK_VIEW.value
    ],
    RoleEnum.GUEST: [
        PermissionEnum.DATA_VIEW.value
    ]
}

# ------------------------------ 数据模型（Pydantic） ------------------------------
if PYDANTIC_LOADED:
    class UserModel(BaseModel):
        """用户模型"""
        username: str = Field(..., description="用户名")
        password_hash: str = Field(..., description="密码哈希")
        email: Optional[EmailStr] = Field(None, description="邮箱")
        full_name: Optional[str] = Field(None, description="全名")
        role: RoleEnum = Field(RoleEnum.GUEST, description="角色")
        is_active: bool = Field(True, description="是否激活")
        created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")
        last_login: Optional[str] = Field(None, description="最后登录时间")
        
        @validator("username")
        def username_validator(cls, v):
            """用户名验证"""
            if not 3 <= len(v) <= 20:
                raise ValueError("用户名长度必须在3-20之间")
            if not v.isalnum() and "_" not in v:
                raise ValueError("用户名只能包含字母、数字和下划线")
            return v
    
    class APIKeyModel(BaseModel):
        """API密钥模型"""
        key_id: str = Field(..., description="密钥ID")
        key_hash: str = Field(..., description="密钥哈希")
        user_id: str = Field(..., description="所属用户")
        name: str = Field(..., description="密钥名称")
        role: RoleEnum = Field(RoleEnum.GUEST, description="密钥权限角色")
        expires_at: Optional[str] = Field(None, description="过期时间（ISO格式）")
        is_active: bool = Field(True, description="是否激活")
        created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")
        last_used: Optional[str] = Field(None, description="最后使用时间")
    
    class TokenModel(BaseModel):
        """令牌模型"""
        access_token: str = Field(..., description="访问令牌")
        refresh_token: str = Field(..., description="刷新令牌")
        token_type: str = Field("bearer", description="令牌类型")
        expires_at: str = Field(..., description="过期时间")
        role: RoleEnum = Field(..., description="令牌权限角色")
    
    class LoginRequest(BaseModel):
        """登录请求模型"""
        username: str = Field(..., description="用户名")
        password: str = Field(..., description="密码")
else:
    # 降级处理：简单类定义
    class UserModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class APIKeyModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class TokenModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class LoginRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

# ------------------------------ 核心认证类 ------------------------------
class ExtIdentityManager:
    """外部身份认证管理器（核心：JWT+RBAC+API密钥）"""
    def __init__(self):
        """初始化认证管理器"""
        # 初始化存储
        self._init_storage()
        
        # 加载数据
        self.users: Dict[str, UserModel] = self._load_users()
        self.api_keys: Dict[str, APIKeyModel] = self._load_api_keys()
        self.permissions: Dict[str, List[str]] = self._load_permissions()
        
        # FastAPI安全工具（按需初始化）
        if FASTAPI_STREAMLIT_LOADED:
            self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
            self.api_key_header = APIKeyHeader(name=IDENTITY_CONFIG["API_KEY_HEADER_NAME"], auto_error=False)
            self.api_key_query = APIKeyQuery(name=IDENTITY_CONFIG["API_KEY_QUERY_NAME"], auto_error=False)

    def _init_storage(self):
        """初始化存储文件"""
        # 初始化用户文件
        if not os.path.exists(USERS_FILE):
            # 创建默认管理员用户（用户名：admin，密码：admin123）
            default_admin = {
                "username": "admin",
                "password_hash": self._hash_password("admin123"),
                "email": "admin@umc-agent.com",
                "full_name": "UMC Admin",
                "role": RoleEnum.ADMIN.value,
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump({"admin": default_admin}, f, ensure_ascii=False, indent=2)
            logger.info("初始化默认管理员用户：admin / admin123（请及时修改密码）")
        
        # 初始化API密钥文件
        if not os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
        # 初始化权限文件
        if not os.path.exists(PERMISSIONS_FILE):
            with open(PERMISSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(ROLE_PERMISSIONS, f, ensure_ascii=False, indent=2)

    # ------------------------------ 密码处理 ------------------------------
    def _hash_password(self, password: str) -> str:
        """密码哈希（bcrypt）"""
        salt = bcrypt.gensalt()
        password_bytes = password.encode("utf-8")
        hash_bytes = bcrypt.hashpw(password_bytes, salt)
        return hash_bytes.decode("utf-8")

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        try:
            password_bytes = password.encode("utf-8")
            hash_bytes = password_hash.encode("utf-8")
            return bcrypt.checkpw(password_bytes, hash_bytes)
        except Exception as e:
            logger.error(f"密码验证失败：{e}")
            return False

    # ------------------------------ JWT令牌处理 ------------------------------
    def _create_tokens(self, username: str, role: RoleEnum) -> TokenModel:
        """生成访问令牌和刷新令牌"""
        # 计算过期时间
        access_expires = datetime.utcnow() + timedelta(minutes=IDENTITY_CONFIG["JWT_ACCESS_TOKEN_EXPIRE_MINUTES"])
        refresh_expires = datetime.utcnow() + timedelta(days=IDENTITY_CONFIG["JWT_REFRESH_TOKEN_EXPIRE_DAYS"])
        
        # 访问令牌载荷
        access_payload = {
            "sub": username,
            "type": "access",
            "role": role.value,
            "exp": access_expires,
            "iat": datetime.utcnow()
        }
        
        # 刷新令牌载荷
        refresh_payload = {
            "sub": username,
            "type": "refresh",
            "role": role.value,
            "exp": refresh_expires,
            "iat": datetime.utcnow()
        }
        
        # 生成令牌
        access_token = jwt.encode(
            access_payload,
            IDENTITY_CONFIG["JWT_SECRET_KEY"],
            algorithm=IDENTITY_CONFIG["JWT_ALGORITHM"]
        )
        refresh_token = jwt.encode(
            refresh_payload,
            IDENTITY_CONFIG["JWT_SECRET_KEY"],
            algorithm=IDENTITY_CONFIG["JWT_ALGORITHM"]
        )
        
        # 返回令牌模型
        return TokenModel(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_at=access_expires.isoformat(),
            role=role
        )

    def _verify_token(self, token: str, token_type: Literal["access", "refresh"] = "access") -> Dict[str, Any]:
        """验证令牌"""
        try:
            # 解码令牌
            payload = jwt.decode(
                token,
                IDENTITY_CONFIG["JWT_SECRET_KEY"],
                algorithms=[IDENTITY_CONFIG["JWT_ALGORITHM"]],
                options={"verify_exp": True}
            )
            
            # 验证令牌类型
            if payload.get("type") != token_type:
                raise ValueError(f"无效的令牌类型，期望：{token_type}")
            
            # 验证用户存在且激活
            username = payload.get("sub")
            if username not in self.users or not self.users[username].is_active:
                raise ValueError("用户不存在或已禁用")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="令牌已过期") if FASTAPI_STREAMLIT_LOADED else ValueError("令牌已过期")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"无效的令牌：{str(e)}") if FASTAPI_STREAMLIT_LOADED else ValueError(f"无效的令牌：{str(e)}")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"令牌验证失败：{str(e)}") if FASTAPI_STREAMLIT_LOADED else ValueError(f"令牌验证失败：{str(e)}")

    # ------------------------------ 用户管理 ------------------------------
    def _load_users(self) -> Dict[str, UserModel]:
        """加载用户数据"""
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users_data = json.load(f)
            
            users = {}
            for username, user_data in users_data.items():
                users[username] = UserModel(**user_data)
            
            return users
        except Exception as e:
            logger.error(f"加载用户数据失败：{e}")
            return {}

    def _save_users(self):
        """保存用户数据"""
        try:
            users_data = {username: user.__dict__ for username, user in self.users.items()}
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存用户数据失败：{e}")
            raise

    def create_user(self, username: str, password: str, email: Optional[str] = None, 
                   full_name: Optional[str] = None, role: RoleEnum = RoleEnum.GUEST) -> UserModel:
        """创建用户"""
        if username in self.users:
            raise ValueError(f"用户 {username} 已存在")
        
        # 创建用户
        user_data = {
            "username": username,
            "password_hash": self._hash_password(password),
            "email": email,
            "full_name": full_name,
            "role": role,
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        user = UserModel(**user_data) if PYDANTIC_LOADED else UserModel(**user_data)
        self.users[username] = user
        self._save_users()
        
        logger.info(f"创建用户：{username}（角色：{role.value}）")
        return user

    def update_user(self, username: str, **kwargs) -> UserModel:
        """更新用户信息"""
        if username not in self.users:
            raise ValueError(f"用户 {username} 不存在")
        
        user = self.users[username]
        
        # 更新字段（仅允许更新指定字段）
        allowed_fields = ["email", "full_name", "role", "is_active", "password_hash"]
        for key, value in kwargs.items():
            if key in allowed_fields and hasattr(user, key):
                setattr(user, key, value)
        
        self._save_users()
        logger.info(f"更新用户：{username}")
        return user

    def delete_user(self, username: str):
        """删除用户"""
        if username not in self.users:
            raise ValueError(f"用户 {username} 不存在")
        
        del self.users[username]
        self._save_users()
        logger.info(f"删除用户：{username}")

    def authenticate_user(self, username: str, password: str) -> Optional[UserModel]:
        """用户认证"""
        if username not in self.users or not self.users[username].is_active:
            return None
        
        user = self.users[username]
        if self._verify_password(password, user.password_hash):
            # 更新最后登录时间
            user.last_login = datetime.now().isoformat()
            self._save_users()
            logger.info(f"用户登录成功：{username}")
            return user
        
        logger.warning(f"用户登录失败：{username}（密码错误）")
        return None

    # ------------------------------ API密钥管理 ------------------------------
    def _generate_api_key(self) -> str:
        """生成API密钥（32位随机字符串）"""
        import secrets
        return secrets.token_hex(16)  # 16字节=32位十六进制字符串

    def _load_api_keys(self) -> Dict[str, APIKeyModel]:
        """加载API密钥"""
        try:
            with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                api_keys_data = json.load(f)
            
            api_keys = {}
            for key_id, key_data in api_keys_data.items():
                api_keys[key_id] = APIKeyModel(**key_data) if PYDANTIC_LOADED else APIKeyModel(**key_data)
            
            return api_keys
        except Exception as e:
            logger.error(f"加载API密钥失败：{e}")
            return {}

    def _save_api_keys(self):
        """保存API密钥"""
        try:
            api_keys_data = {key_id: key.__dict__ for key_id, key in self.api_keys.items()}
            with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
                json.dump(api_keys_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存API密钥失败：{e}")
            raise

    def create_api_key(self, user_id: str, name: str, role: RoleEnum = RoleEnum.GUEST, 
                      expires_at: Optional[datetime] = None) -> Tuple[str, APIKeyModel]:
        """创建API密钥（返回原始密钥和密钥模型）"""
        if user_id not in self.users:
            raise ValueError(f"用户 {user_id} 不存在")
        
        # 生成密钥ID和原始密钥
        key_id = f"key_{datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}" if 'secrets' in locals() else f"key_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        raw_key = self._generate_api_key() if 'secrets' in locals() else f"apikey_{secrets.token_hex(16)}"
        
        # 哈希密钥（仅存储哈希）
        key_hash = self._hash_password(raw_key)
        
        # 构建密钥模型
        key_data = {
            "key_id": key_id,
            "key_hash": key_hash,
            "user_id": user_id,
            "name": name,
            "role": role,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "last_used": None
        }
        
        api_key = APIKeyModel(**key_data) if PYDANTIC_LOADED else APIKeyModel(**key_data)
        self.api_keys[key_id] = api_key
        self._save_api_keys()
        
        logger.info(f"创建API密钥：{key_id}（用户：{user_id}，角色：{role.value}）")
        return raw_key, api_key

    def verify_api_key(self, api_key: str) -> Optional[APIKeyModel]:
        """验证API密钥"""
        # 遍历所有API密钥
        for key_id, key_model in self.api_keys.items():
            # 检查是否激活
            if not key_model.is_active:
                continue
            
            # 检查是否过期
            if key_model.expires_at:
                expires_dt = datetime.fromisoformat(key_model.expires_at)
                if expires_dt < datetime.now():
                    continue
            
            # 验证密钥哈希
            if self._verify_password(api_key, key_model.key_hash):
                # 更新最后使用时间
                key_model.last_used = datetime.now().isoformat()
                self._save_api_keys()
                logger.info(f"API密钥验证成功：{key_id}（用户：{key_model.user_id}）")
                return key_model
        
        logger.warning("API密钥验证失败：无效的密钥")
        return None

    # ------------------------------ 权限管理 ------------------------------
    def _load_permissions(self) -> Dict[str, List[str]]:
        """加载权限配置"""
        try:
            with open(PERMISSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载权限配置失败：{e}")
            return ROLE_PERMISSIONS

    def has_permission(self, role: Union[RoleEnum, str], permission: Union[PermissionEnum, str]) -> bool:
        """检查角色是否拥有指定权限"""
        # 统一格式
        role_str = role.value if isinstance(role, RoleEnum) else role
        perm_str = permission.value if isinstance(permission, PermissionEnum) else permission
        
        # 获取角色的所有权限
        role_perms = self.permissions.get(role_str, [])
        
        # 检查权限
        return perm_str in role_perms

    # ------------------------------ FastAPI中间件 ------------------------------
    def get_current_user(self, request: Request = None, token: str = Depends(None)) -> Dict[str, Any]:
        """FastAPI依赖：获取当前用户（JWT令牌）"""
        if not FASTAPI_STREAMLIT_LOADED:
            raise RuntimeError("FastAPI未安装，无法使用该依赖")
        
        # 验证令牌
        payload = self._verify_token(token)
        username = payload.get("sub")
        role = payload.get("role")
        
        # 返回用户信息
        return {
            "username": username,
            "role": role,
            "permissions": self.permissions.get(role, [])
        }

    def require_permission(self, required_permission: PermissionEnum):
        """FastAPI依赖：权限校验装饰器"""
        def dependency(current_user: Dict = Depends(self.get_current_user)):
            if not self.has_permission(current_user["role"], required_permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"权限不足：需要 {required_permission.value} 权限"
                )
            return current_user
        return dependency

    def authenticate_request(self, request: Request) -> Dict[str, Any]:
        """认证请求（支持JWT令牌或API密钥）"""
        if not FASTAPI_STREAMLIT_LOADED:
            raise RuntimeError("FastAPI未安装，无法认证请求")
        
        # 尝试JWT令牌认证
        try:
            token = self.oauth2_scheme(request)
            payload = self._verify_token(token)
            return {
                "type": "jwt",
                "username": payload.get("sub"),
                "role": payload.get("role"),
                "permissions": self.permissions.get(payload.get("role"), [])
            }
        except Exception:
            pass
        
        # 尝试API密钥认证
        api_key = None
        # 从请求头获取
        if self.api_key_header:
            api_key = self.api_key_header(request)
        # 从URL参数获取
        if not api_key and self.api_key_query:
            api_key = self.api_key_query(request)
        
        if api_key:
            key_model = self.verify_api_key(api_key)
            if key_model:
                return {
                    "type": "api_key",
                    "key_id": key_model.key_id,
                    "user_id": key_model.user_id,
                    "role": key_model.role.value if isinstance(key_model.role, RoleEnum) else key_model.role,
                    "permissions": self.permissions.get(key_model.role, [])
                }
        
        # 认证失败
        raise HTTPException(
            status_code=401,
            detail="认证失败：请提供有效的JWT令牌或API密钥",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # ------------------------------ Streamlit仪表盘集成 ------------------------------
    def streamlit_login_widget(self) -> Optional[UserModel]:
        """Streamlit登录组件"""
        if not FASTAPI_STREAMLIT_LOADED:
            raise RuntimeError("Streamlit未安装，无法使用登录组件")
        
        # 检查会话状态
        if "authenticated" in st.session_state and st.session_state.authenticated:
            return st.session_state.user
        
        # 登录表单
        st.subheader("🔐 UMC智能体身份认证")
        
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            login_btn = st.form_submit_button("登录", type="primary")
            
            if login_btn:
                if not username or not password:
                    st.error("请输入用户名和密码")
                else:
                    # 认证用户
                    user = self.authenticate_user(username, password)
                    if user:
                        # 设置会话状态
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.role = user.role
                        st.session_state.permissions = self.permissions.get(user.role.value, [])
                        
                        st.success(f"✅ 登录成功！欢迎 {user.full_name or user.username}")
                        st.rerun()
                    else:
                        st.error("❌ 用户名或密码错误")
        
        return None

    def streamlit_check_permission(self, permission: PermissionEnum) -> bool:
        """Streamlit权限检查"""
        if not FASTAPI_STREAMLIT_LOADED:
            return False
        
        if not st.session_state.get("authenticated", False):
            return False
        
        return self.has_permission(st.session_state.role, permission)

    # ------------------------------ 快捷方法 ------------------------------
    def login(self, username: str, password: str) -> Optional[TokenModel]:
        """用户登录（生成令牌）"""
        user = self.authenticate_user(username, password)
        if user:
            return self._create_tokens(username, user.role)
        return None

    def refresh_token(self, refresh_token: str) -> Optional[TokenModel]:
        """刷新访问令牌"""
        try:
            # 验证刷新令牌
            payload = self._verify_token(refresh_token, token_type="refresh")
            username = payload.get("sub")
            role = RoleEnum(payload.get("role"))
            
            # 生成新令牌
            return self._create_tokens(username, role)
        except Exception as e:
            logger.error(f"刷新令牌失败：{e}")
            return None

# ------------------------------ 快捷使用函数 ------------------------------
def create_identity_manager() -> ExtIdentityManager:
    """创建身份认证管理器实例"""
    return ExtIdentityManager()

def init_default_admin(password: str = "admin123"):
    """初始化默认管理员（重置密码）"""
    identity = create_identity_manager()
    
    # 更新管理员密码
    if "admin" in identity.users:
        identity.update_user(
            "admin",
            password_hash=identity._hash_password(password)
        )
        logger.info("默认管理员密码已重置")
    else:
        # 创建管理员
        identity.create_user(
            username="admin",
            password=password,
            email="admin@umc-agent.com",
            full_name="UMC Admin",
            role=RoleEnum.ADMIN
        )
    
    print(f"默认管理员初始化完成：")
    print(f"  用户名：admin")
    print(f"  密码：{password}")
    print(f"  请及时修改默认密码！")

# ------------------------------ 命令行入口 ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UMC智能体身份认证工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", required=True, help="操作命令")
    
    # 初始化管理员
    parser_init = subparsers.add_parser("init-admin", help="初始化默认管理员")
    parser_init.add_argument("--password", "-p", type=str, default="admin123", help="管理员密码")
    
    # 创建用户
    parser_create = subparsers.add_parser("create-user", help="创建用户")
    parser_create.add_argument("--username", "-u", type=str, required=True, help="用户名")
    parser_create.add_argument("--password", "-p", type=str, required=True, help="密码")
    parser_create.add_argument("--role", "-r", type=str, default="guest", 
                              choices=["admin", "operator", "viewer", "guest"], help="角色")
    parser_create.add_argument("--email", "-e", type=str, help="邮箱")
    parser_create.add_argument("--full-name", "-n", type=str, help="全名")
    
    # 创建API密钥
    parser_apikey = subparsers.add_parser("create-api-key", help="创建API密钥")
    parser_apikey.add_argument("--user-id", "-u", type=str, required=True, help="用户ID")
    parser_apikey.add_argument("--name", "-n", type=str, required=True, help="密钥名称")
    parser_apikey.add_argument("--role", "-r", type=str, default="guest", 
                              choices=["admin", "operator", "viewer", "guest"], help="密钥角色")
    parser_apikey.add_argument("--expires-days", "-d", type=int, help="过期天数")
    
    # 解析参数
    args = parser.parse_args()
    identity = create_identity_manager()
    
    # 执行命令
    if args.command == "init-admin":
        init_default_admin(args.password)
    
    elif args.command == "create-user":
        try:
            role = RoleEnum(args.role)
            user = identity.create_user(
                username=args.username,
                password=args.password,
                email=args.email,
                full_name=args.full_name,
                role=role
            )
            print(f"✅ 用户创建成功：{args.username}（角色：{args.role}）")
        except Exception as e:
            print(f"❌ 创建用户失败：{e}")
    
    elif args.command == "create-api-key":
        try:
            # 计算过期时间
            expires_at = None
            if args.expires_days:
                expires_at = datetime.now() + timedelta(days=args.expires_days)
            
            # 创建API密钥
            raw_key, api_key = identity.create_api_key(
                user_id=args.user_id,
                name=args.name,
                role=RoleEnum(args.role),
                expires_at=expires_at
            )
            
            print(f"✅ API密钥创建成功：")
            print(f"  密钥ID：{api_key.key_id}")
            print(f"  原始密钥：{raw_key}（请妥善保存，仅显示一次！）")
            print(f"  所属用户：{api_key.user_id}")
            print(f"  角色：{api_key.role}")
            print(f"  创建时间：{api_key.created_at}")
            if api_key.expires_at:
                print(f"  过期时间：{api_key.expires_at}")
        except Exception as e:
            print(f"❌ 创建API密钥失败：{e}")