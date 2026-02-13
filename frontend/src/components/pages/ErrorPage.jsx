import React from 'react';

const ErrorPage = ({ error }) => {
    // Parse URL parameters for dynamic error messages
    const params = new URLSearchParams(window.location.search);
    const msg = params.get('message');
    const code = params.get('code');

    const resolveMessage = () => {
        if (error) {
            return typeof error === 'string' ? error : error.message || 'An unexpected error occurred.';
        }
        if (msg) return decodeURIComponent(msg);
        return 'An unexpected error occurred.';
    };

    const resolveCode = () => {
        if (error && error.code) return error.code;
        if (code) return code;
        return 'ERROR';
    };

    const errorMessage = resolveMessage();
    const errorCode = resolveCode();

    // Inner styles to match static/error.html
    const styles = {
        container: {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100vh',
            textAlign: 'center',
            padding: '2rem'
        },
        code: {
            fontFamily: "'Space Mono', monospace",
            fontSize: '4rem',
            color: '#ef5350',
            marginBottom: '1rem'
        },
        message: {
            fontSize: '1.5rem',
            marginBottom: '2rem',
            marginBottom: '2rem',
            color: '#e0e0e0' // High contrast text
        },
        btn: {
            padding: '1rem 2rem',
            background: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            color: 'white',
            textDecoration: 'none',
            borderRadius: '8px',
            fontFamily: "'Space Mono', monospace",
            transition: 'all 0.3s ease',
            cursor: 'pointer'
        }
    };

    return (
        <div className="app-container">
            <div style={styles.container}>
                <div style={styles.code}>{errorCode}</div>
                <div style={styles.message}>{errorMessage}</div>
                <a href="/" style={styles.btn} className="home-btn-hover">
                    RETURN TO DASHBOARD
                </a>
            </div>
            <style>{`
                .home-btn-hover:hover {
                    background: rgba(255, 255, 255, 0.2) !important;
                    border-color: white !important;
                }
            `}</style>
        </div>
    );
};

export default ErrorPage;
