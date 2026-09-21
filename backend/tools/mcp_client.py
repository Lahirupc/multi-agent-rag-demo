from langsmith import traceable
from logger import logger

class MockMCPServer:
    """
    Simulates a Model Context Protocol (MCP) server that exposes enterprise data
    like Employee Directories, Service Catalogs, or active incident IT Service Management (ITSM) systems.
    """
    
    def __init__(self):
        self.employee_directory = {
            "john.smith": {"department": "Payments", "title": "Senior Engineer"},
            "alice.jones": {"department": "Security", "title": "CISO"}
        }
        
        self.service_catalog = {
            "pay-gateway": {"status": "degraded", "owner": "john.smith", "upstream": "auth-service"},
            "auth-service": {"status": "active", "owner": "alice.jones", "upstream": "none"}
        }

    @traceable(run_type="tool", name="mcp_employee_search")
    async def get_employee_info(self, username: str) -> dict:
        """Fetch employee metadata from the enterprise Active Directory (Mock)."""
        logger.info("mcp_tool_invoked", tool="employee_search", username=username)
        return self.employee_directory.get(username.lower(), {"error": "Employee not found."})

    @traceable(run_type="tool", name="mcp_catalog_search")
    async def get_service_health(self, service_name: str) -> dict:
        """Fetch real-time microservice health from the internal developer portal (Mock)."""
        logger.info("mcp_tool_invoked", tool="catalog_search", service_name=service_name)
        return self.service_catalog.get(service_name.lower(), {"error": "Service not registered in catalog."})
