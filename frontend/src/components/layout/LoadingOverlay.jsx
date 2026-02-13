import React, { useEffect } from 'react';
import { useStock } from '../../context/StockContext';

const LoadingOverlay = () => {
    const { loading, loadingStatus } = useStock();

    // Hide body scrollbar when loading
    useEffect(() => {
        if (loading) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        
        return () => {
            document.body.style.overflow = '';
        };
    }, [loading]);

    if (!loading) return null;

    return (
        <>
            <style>{`
                @keyframes slideLoadingBar {
                    0% {
                        transform: translateX(-100%);
                    }
                    100% {
                        transform: translateX(350%);
                    }
                }
            `}</style>
            <div style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: '#000000',
                zIndex: 99999,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '1.5rem'
            }}>
                {/* Big Visible Loading Bar */}
                <div style={{
                    width: '400px',
                    height: '4px',
                    backgroundColor: '#222',
                    borderRadius: '2px',
                    overflow: 'hidden'
                }}>
                    <div style={{
                        width: '40%',
                        height: '100%',
                        backgroundColor: '#fff',
                        animation: 'slideLoadingBar 1.5s ease-in-out infinite',
                        boxShadow: '0 0 10px #fff'
                    }}></div>
                </div>

                {/* Status Text */}
                <div style={{
                    color: '#fff',
                    fontSize: '14px',
                    fontFamily: 'monospace',
                    letterSpacing: '2px',
                    textTransform: 'uppercase'
                }}>
                    {loadingStatus || 'INITIALIZING...'}
                </div>
            </div>
        </>
    );
};

export default LoadingOverlay;
