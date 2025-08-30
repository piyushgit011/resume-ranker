import streamlit as st
import hashlib
import sqlite3
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import secrets

class AuthService:
    """Authentication and user management service"""
    
    def __init__(self, db_path: str = "/workspace/data/resume_ranker.db"):
        self.db_path = db_path
        self._initialize_auth_tables()
    
    def _initialize_auth_tables(self):
        """Initialize authentication tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    role TEXT DEFAULT 'user',  -- 'admin', 'hr_manager', 'user'
                    company TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    permission TEXT NOT NULL,  -- 'view_analytics', 'export_data', 'manage_users'
                    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Create default admin user if none exists
            self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if no users exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            if user_count == 0:
                # Create default admin
                admin_password = "admin123"  # Should be changed immediately
                password_hash = self._hash_password(admin_password)
                
                conn.execute("""
                    INSERT INTO users (username, email, password_hash, full_name, role)
                    VALUES (?, ?, ?, ?, ?)
                """, ("admin", "admin@company.com", password_hash, "System Administrator", "admin"))
                
                print("Default admin user created: admin/admin123 (CHANGE PASSWORD IMMEDIATELY)")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user data"""
        password_hash = self._hash_password(password)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM users 
                WHERE username = ? AND password_hash = ? AND is_active = TRUE
            """, (username, password_hash))
            
            user = cursor.fetchone()
            
            if user:
                # Update last login
                conn.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                """, (user['id'],))
                
                return dict(user)
            
            return None
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: str, role: str = "user", company: str = "") -> bool:
        """Create new user"""
        password_hash = self._hash_password(password)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (username, email, password_hash, full_name, role, company)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (username, email, password_hash, full_name, role, company))
                return True
        except sqlite3.IntegrityError:
            return False  # User already exists
    
    def create_session(self, user_id: int) -> str:
        """Create user session and return session token"""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)  # 7 day session
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            """, (user_id, session_token, expires_at))
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT u.*, s.expires_at
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND s.is_active = TRUE AND s.expires_at > CURRENT_TIMESTAMP
            """, (session_token,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def logout_user(self, session_token: str):
        """Logout user by deactivating session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE user_sessions SET is_active = FALSE WHERE session_token = ?
            """, (session_token,))
    
    def get_user_permissions(self, user_id: int) -> List[str]:
        """Get user permissions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT permission FROM user_permissions WHERE user_id = ?
            """, (user_id,))
            
            return [row[0] for row in cursor.fetchall()]
    
    def has_permission(self, user_id: int, permission: str) -> bool:
        """Check if user has specific permission"""
        permissions = self.get_user_permissions(user_id)
        return permission in permissions
    
    def grant_permission(self, user_id: int, permission: str):
        """Grant permission to user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO user_permissions (user_id, permission)
                VALUES (?, ?)
            """, (user_id, permission))

def require_auth(func):
    """Decorator to require authentication for Streamlit functions"""
    def wrapper(*args, **kwargs):
        if 'authenticated' not in st.session_state or not st.session_state.authenticated:
            st.error("🔒 Please log in to access this feature")
            return None
        return func(*args, **kwargs)
    return wrapper

def show_login_form(auth_service: AuthService):
    """Show login form"""
    st.markdown("### 🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user = auth_service.authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.session_token = auth_service.create_session(user['id'])
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

def show_user_management(auth_service: AuthService):
    """Show user management interface (admin only)"""
    if not st.session_state.get('user', {}).get('role') == 'admin':
        st.error("🚫 Admin access required")
        return
    
    st.markdown("### 👥 User Management")
    
    # Create new user
    with st.expander("➕ Create New User"):
        with st.form("create_user_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            new_full_name = st.text_input("Full Name")
            new_role = st.selectbox("Role", ["user", "hr_manager", "admin"])
            new_company = st.text_input("Company")
            
            if st.form_submit_button("Create User"):
                if auth_service.create_user(new_username, new_email, new_password, 
                                         new_full_name, new_role, new_company):
                    st.success("✅ User created successfully!")
                else:
                    st.error("❌ Failed to create user (username/email may already exist)")

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user"""
    return st.session_state.get('user')