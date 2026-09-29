import pytest


def test_download_verification_rejects_login_html_and_untrusted_hosts():
    from thermis.nasa_access import trusted_download, verify_bytes

    assert verify_bytes(b"II*\x00" + b"0" * 32, "image/tiff")
    assert not verify_bytes(b"<html>Login</html>", "text/html")
    assert not verify_bytes(b"", "application/octet-stream")
    assert trusted_download("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/a.tif")
    assert not trusted_download("https://evil.test/a.tif")
    assert not trusted_download("http://data.lpdaac.earthdatacloud.nasa.gov/a.tif")


def test_invalid_token_response_is_not_persisted(tmp_path):
    from thermis.nasa_access import persist_token

    path = tmp_path / ".env"
    with pytest.raises(ValueError):
        persist_token({}, path)
    assert not path.exists()


def test_protected_redirect_does_not_forward_earthdata_token():
    import httpx

    from thermis.nasa_access import verify_download

    def handler(request):
        if request.url.host == "data.lpdaac.earthdatacloud.nasa.gov":
            assert request.headers["authorization"] == "Bearer test-token"
            return httpx.Response(
                303, headers={"Location": "https://cdn.cloudfront.net/test.tif?signature=private"}
            )
        assert "authorization" not in request.headers
        return httpx.Response(
            206, headers={"Content-Type": "image/tiff"}, content=b"II*\x00" + b"0" * 4092
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = verify_download(
            client,
            "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/a.tif",
            "test-token",
        )
    assert report["protected_download_verified"]
    assert report["bytes_verified"] == 4096
    assert "signature" not in str(report)
