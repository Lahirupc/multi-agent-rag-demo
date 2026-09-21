from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

security = HTTPBearer()

class User(BaseModel):
    username: str
    roles: List[str]

# Option A: Hardcoded users and roles
HARDCODED_USERS = {
    "token_viewer": User(username="viewer_user", roles=["Viewer"]),
    "token_analyst": User(username="analyst_user", roles=["Analyst"]),
    "token_admin": User(username="admin_user", roles=["Administrator"])
}

ROLE_PERMISSIONS = {
    "Viewer": ["chat", "search"],
    "Analyst": ["chat", "search", "analytics_tools", "mcp_tools"],
    "Administrator": ["chat", "search", "analytics_tools", "mcp_tools", "administrative_tools"]
}

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    token = credentials.credentials
    user = HARDCODED_USERS.get(token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials or Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_active_user(current_user: User = Security(get_current_user)) -> User:
    # Here you could check if the user is disabled
    return current_user

def check_permission(user: User, required_permission: str):
    user_permissions = set()
    for role in user.roles:
        perms = ROLE_PERMISSIONS.get(role, [])
        user_permissions.update(perms)
        
    if required_permission not in user_permissions:
        raise HTTPException(status_code=403, detail="Not enough permissions to perform this action")
