from backend.app.services.security.upload_validator import AdvancedDocumentValidator


def test_sanitize_filename_strips_slashes():
    forward = AdvancedDocumentValidator.sanitize_filename("folder/file.txt")
    assert "/" not in forward
    backward = AdvancedDocumentValidator.sanitize_filename("folder\\file.txt")
    assert "\\" not in backward


def test_sanitize_filename_leading_trailing_dots_spaces():
    result = AdvancedDocumentValidator.sanitize_filename("  ..hidden..  ")
    assert not result.startswith(".")
    assert not result.endswith(".")
    assert " " not in result[0] if result else True


def test_sanitize_filename_all_control_chars():
    result = AdvancedDocumentValidator.sanitize_filename("\x00\x01\x02bad\x7f.txt")
    assert "\x00" not in result
    assert "\x01" not in result
    assert "\x02" not in result
    assert "\x7f" not in result


def test_sanitize_filename_exactly_255_chars():
    name = "a" * 250 + ".txt"
    result = AdvancedDocumentValidator.sanitize_filename(name)
    assert len(result) <= 255


def test_sanitize_filename_reserved_names():
    for reserved in ["con", "prn", "aux", "nul", "com1", "lpt1"]:
        result = AdvancedDocumentValidator.sanitize_filename(f"{reserved}.txt")
        # Should not be exactly the reserved name
        assert result.lower() not in ("con.txt", "prn.txt", "aux.txt", "nul.txt", "com1.txt", "lpt1.txt")
