# utils/auth.py - Simplified for 2 roles only
import streamlit as st
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Simplified user roles - only 2 types
USER_ROLES = {
    "business_owner": {
        "name": "Business Owner",
        "permissions": [
            "view_all_stores",
            "view_all_analytics", 
            "access_setup",
            "access_transform",
            "access_dashboard"
        ],
        "store_access": "all"
    },
    "store_manager": {
        "name": "Store Manager", 
        "permissions": [
            "view_own_store",
            "view_store_analytics",
            "access_dashboard"
        ],
        "store_access": "own"
    }
}

class AuthManager:
    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.users = self.load_users()
        
    def load_users(self) -> Dict:
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading users: {e}")
                return {}
        return self.create_default_users()
    
    def create_default_users(self) -> Dict:
        """Create default users for demo purposes"""
        default_users = {
            "admin": {
                "password": self.hash_password("admin123"),
                "role": "business_owner",
                "name": "Business Owner",
                "email": "owner@retailcompany.com",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "store_ids": [],
                "is_active": True
            },
            "manager_tunis": {
                "password": self.hash_password("manager123"),
                "role": "store_manager", 
                "name": "Ahmed Ben Ali",
                "email": "ahmed.benali@retailcompany.com",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "store_ids": ["STORE_001"],  # Tunis store
                "is_active": True
            },
            "manager_sfax": {
                "password": self.hash_password("manager123"),
                "role": "store_manager",
                "name": "Fatma Trabelsi", 
                "email": "fatma.trabelsi@retailcompany.com",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "store_ids": ["STORE_002"],  # Sfax store
                "is_active": True
            }
        }
        self.save_users(default_users)
        return default_users
    
    def save_users(self, users: Dict):
        """Save users to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            self.users = users
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.hash_password(password) == hashed
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """Authenticate user credentials"""
        if username not in self.users:
            return False, None
        
        user = self.users[username]
        if not user.get("is_active", True):
            return False, None
        
        if self.verify_password(password, user["password"]):
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            self.users[username] = user
            self.save_users(self.users)
            
            return True, {
                "username": username,
                "role": user["role"],
                "name": user["name"],
                "email": user["email"],
                "store_ids": user.get("store_ids", []),
                "permissions": USER_ROLES[user["role"]]["permissions"]
            }
        
        return False, None
    
    def get_accessible_stores(self, username: str, all_stores: List[str]) -> List[str]:
        """Get stores accessible by user"""
        if username not in self.users:
            return []
        
        user = self.users[username]
        role = user.get("role", "")
        store_access = USER_ROLES.get(role, {}).get("store_access", "none")
        
        if store_access == "all":
            return all_stores
        elif store_access == "own":
            return user.get("store_ids", [])
        
        return []

def init_auth():
    """Initialize authentication system"""
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

def show_login_page():
    """Display login page"""
    
    
    # Custom CSS for login page
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background: white;
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .demo-credentials {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 4px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class='login-header'>
        <h1>🏪 Retail Analytics</h1>
        <p>Professional Business Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form
    with st.form("login_form"):
        st.subheader("Sign In")
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("🔐 Sign In", type="primary", use_container_width=True)
        with col2:
            demo_button = st.form_submit_button("👀 Demo Mode", use_container_width=True)
        
        if login_button:
            if username and password:
                auth_success, user_info = st.session_state.auth_manager.authenticate(username, password)
                
                if auth_success:
                    st.session_state.authenticated = True
                    st.session_state.user_info = user_info
                    st.success(f"Welcome back, {user_info['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.error("❌ Please enter both username and password")
        
        if demo_button:
            # Auto-login as business owner for demo
            auth_success, user_info = st.session_state.auth_manager.authenticate("admin", "admin123")
            if auth_success:
                st.session_state.authenticated = True
                st.session_state.user_info = user_info
                st.success("🎭 Demo mode activated - logged in as Business Owner")
                st.rerun()
    
    # Demo credentials
    st.markdown("""
    <div class='demo-credentials'>
        <h4>🎯 Demo Credentials</h4>
        <p><strong>Business Owner:</strong> admin / admin123</p>
        <p><strong>Store Manager (Tunis):</strong> manager_tunis / manager123</p>
        <p><strong>Store Manager (Sfax):</strong> manager_sfax / manager123</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_user_profile():
    """Display user profile in sidebar"""
    if not st.session_state.get('authenticated'):
        return
    
    user_info = st.session_state.user_info
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Profile")
        
        # User info
        st.write(f"**{user_info['name']}**")
        st.write(f"*{USER_ROLES[user_info['role']]['name']}*")
        st.caption(f"📧 {user_info['email']}")
        
        # Store access info
        if user_info['store_ids']:
            st.write(f"🏪 **Stores:** {', '.join(user_info['store_ids'])}")
        elif user_info['role'] == 'business_owner':
            st.write("🏪 **Access:** All Stores")
        
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            # Clear authentication and user info
            st.session_state.authenticated = False
            st.session_state.user_info = None
            
            # Clear database-related session state
            for key in ['csv_paths', 'source_engine', 'table_mapping', 'column_mappings', 'db_identifier']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear Streamlit cache
            st.cache_data.clear()
            
            # Provide feedback and rerun
            st.success("Logged out successfully! Cache and session cleared.")
            st.rerun()

def check_permission(permission: str) -> bool:
    """Check if current user has permission"""
    if not st.session_state.get('authenticated'):
        return False
    
    user_info = st.session_state.user_info
    return permission in user_info.get('permissions', [])

def get_user_stores() -> List[str]:
    """Get stores accessible by current user"""
    if not st.session_state.get('authenticated'):
        return []
    
    user_info = st.session_state.user_info
    
    # Get all available stores from session state
    all_stores = []
    if st.session_state.get('csv_paths'):
        # Try to extract store IDs from data
        try:
            df_magasin = pd.read_csv(st.session_state.csv_paths.get('Magasin', ''))
            if not df_magasin.empty:
                store_col = st.session_state.get('column_mappings', {}).get('Magasin', {}).get('id_magasin', 'id_magasin')
                if store_col in df_magasin.columns:
                    all_stores = df_magasin[store_col].unique().tolist()
        except:
            pass
    
    # If no stores found in data, use demo store IDs
    if not all_stores:
        all_stores = ["STORE_001", "STORE_002", "STORE_003", "STORE_004", "STORE_005"]
    
    return st.session_state.auth_manager.get_accessible_stores(
        st.session_state.user_info['username'], 
        all_stores
    )

def filter_data_by_user_access(df: pd.DataFrame, store_column: str) -> pd.DataFrame:
    """Filter dataframe based on user's store access"""
    if not st.session_state.get('authenticated'):
        return pd.DataFrame()
    
    user_stores = get_user_stores()
    user_info = st.session_state.user_info
    
    # Business owners see all data
    if user_info['role'] == 'business_owner':
        return df
    
    # Filter for specific stores
    if store_column in df.columns and user_stores:
        return df[df[store_column].isin(user_stores)]
    
    return df