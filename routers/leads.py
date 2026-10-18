from fastapi import APIRouter

# Note: This file will be expanded with CRUD operations and lead-specific logic in later phases.
router = APIRouter(prefix="/leads", tags=["Leads"])

@router.get("/test")
def test_router():
    """
    Test endpoint to confirm the leads router is properly connected.
    """
    return {"message": "Leads router is working correctly"}
