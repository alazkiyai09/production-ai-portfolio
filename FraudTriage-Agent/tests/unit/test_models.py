"""
Unit tests for data models.

Tests Pydantic models for validation and serialization.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models.alert import Alert, AlertStatus, AlertType, FraudAlert, Transaction, TransactionType
from src.models.agent import AgentState, RiskAssessment, TriageResult
from src.models.review import HumanReview, HumanReviewDecision, ReviewRequest


class TestTransactionModel:
    """Test Transaction model."""

    def test_create_transaction(self):
        """Test creating a valid transaction."""
        txn = Transaction(
            transaction_id="txn-001",
            account_id="acc-123",
            amount=100.0,
            currency="USD",
            transaction_type=TransactionType.PURCHASE,
            location_country="US",
            timestamp=datetime.now(),
        )
        assert txn.transaction_id == "txn-001"
        assert txn.amount == 100.0
        assert txn.transaction_type == TransactionType.PURCHASE

    def test_transaction_amount_validation(self):
        """Test that amount must be positive."""
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="txn-001",
                account_id="acc-123",
                amount=-10.0,  # Invalid: negative
                currency="USD",
                transaction_type=TransactionType.PURCHASE,
                location_country="US",
                timestamp=datetime.now(),
            )

    def test_transaction_serialization(self):
        """Test transaction JSON serialization."""
        txn = Transaction(
            transaction_id="txn-001",
            account_id="acc-123",
            amount=100.0,
            transaction_type=TransactionType.PURCHASE,
            location_country="US",
            timestamp=datetime(2025, 1, 30, 12, 0, 0),
        )
        data = txn.model_dump()
        assert data["transaction_id"] == "txn-001"
        assert data["amount"] == 100.0


class TestFraudAlertModel:
    """Test FraudAlert model."""

    def test_create_fraud_alert(self):
        """Test creating a valid fraud alert."""
        transaction = Transaction(
            transaction_id="txn-001",
            account_id="acc-123",
            amount=2500.0,
            transaction_type=TransactionType.PURCHASE,
            location_country="NG",
            timestamp=datetime.now(),
        )

        alert = FraudAlert(
            alert_type=AlertType.UNUSUAL_LOCATION,
            account_id="acc-123",
            customer_id="cust-456",
            transaction=transaction,
            alert_reason="Transaction from high-risk country",
        )

        assert alert.alert_type == AlertType.UNUSUAL_LOCATION
        assert alert.status == AlertStatus.PENDING
        assert alert.transaction.amount == 2500.0
        assert alert.requires_human_review is False  # Default

    def test_alert_defaults(self):
        """Test alert default values."""
        transaction = Transaction(
            transaction_id="txn-001",
            account_id="acc-123",
            amount=100.0,
            transaction_type=TransactionType.PURCHASE,
            location_country="US",
            timestamp=datetime.now(),
        )

        alert = FraudAlert(
            alert_type=AlertType.OTHER,
            account_id="acc-123",
            customer_id="cust-456",
            transaction=transaction,
            alert_reason="Test alert",
        )

        assert alert.status == AlertStatus.PENDING
        assert alert.severity == "medium"
        assert alert.resolved is False


class TestRiskAssessment:
    """Test RiskAssessment model."""

    def test_create_risk_assessment(self):
        """Test creating a valid risk assessment."""
        assessment = RiskAssessment(
            risk_score=75,
            risk_factors=["Unusual location", "High amount"],
            reasoning="Multiple risk indicators present",
            confidence=0.85,
            suggested_action="escalate_for_review",
        )

        assert assessment.risk_score == 75
        assert len(assessment.risk_factors) == 2
        assert assessment.confidence == 0.85

    def test_risk_score_bounds(self):
        """Test risk score is bounded 0-100."""
        # Valid scores
        RiskAssessment(
            risk_score=0,
            risk_factors=[],
            reasoning="Test",
            confidence=0.5,
            suggested_action="auto_close",
        )
        RiskAssessment(
            risk_score=100,
            risk_factors=[],
            reasoning="Test",
            confidence=0.5,
            suggested_action="escalate_for_review",
        )

        # Invalid score
        with pytest.raises(ValidationError):
            RiskAssessment(
                risk_score=150,  # Invalid: > 100
                risk_factors=[],
                reasoning="Test",
                confidence=0.5,
                suggested_action="escalate_for_review",
            )


class TestTriageResult:
    """Test TriageResult model."""

    def test_create_triage_result(self):
        """Test creating a valid triage result."""
        result = TriageResult(
            alert_id="alert-001",
            risk_score=65,
            risk_factors=["New device", "Unusual location"],
            confidence=0.8,
            recommendation="Escalate to analyst",
            next_action="escalate_for_review",
            requires_human_review=True,
            timestamp="2025-01-30T12:00:00Z",
        )

        assert result.alert_id == "alert-001"
        assert result.requires_human_review is True
        assert result.next_action == "escalate_for_review"

    def test_triage_result_with_optional_fields(self):
        """Test triage result with optional fields populated."""
        result = TriageResult(
            alert_id="alert-002",
            risk_score=20,
            risk_factors=[],
            confidence=0.9,
            recommendation="Auto-close as legitimate",
            next_action="auto_close",
            requires_human_review=False,
            transaction_summary="3 transactions, all from US",
            customer_summary="Customer: John Doe, 3 year account",
            analysis_duration_ms=1500,
            model_used="glm-4-plus",
            timestamp="2025-01-30T12:00:00Z",
        )

        assert result.transaction_summary is not None
        assert result.customer_summary is not None
        assert result.analysis_duration_ms == 1500


class TestHumanReview:
    """Test HumanReview model."""

    def test_create_human_review(self):
        """Test creating a valid human review."""
        review = HumanReview(
            alert_id="alert-001",
            reviewer_id="analyst-001",
            reviewer_name="Jane Smith",
            decision=HumanReviewDecision.CONFIRM_FRAUD,
            reasoning="Customer confirmed unauthorized transaction",
        )

        assert review.decision == HumanReviewDecision.CONFIRM_FRAUD
        assert review.case_created is False  # Default
        assert review.review_id is not None  # Auto-generated UUID

    def test_review_request(self):
        """Test review request model."""
        request = ReviewRequest(
            reviewer_id="analyst-001",
            reviewer_name="Jane Smith",
            decision=HumanReviewDecision.CONFIRM_LEGITIMATE,
            reasoning="Customer verified transaction",
            agreed_with_agent=True,
            correct_risk_score=15,
        )

        assert request.decision == HumanReviewDecision.CONFIRM_LEGITIMATE
        assert request.agreed_with_agent is True
        assert request.correct_risk_score == 15
