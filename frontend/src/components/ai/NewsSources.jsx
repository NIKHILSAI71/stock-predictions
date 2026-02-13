import React from 'react';
import PropTypes from 'prop-types';

const NewsSources = ({ context }) => {
    if (!context || context.length === 0) return null;

    // Helper to extract domain
    const getDomain = (url) => {
        try {
            return new URL(url).hostname;
        } catch {
            return null;
        }
    };

    const [expanded, setExpanded] = React.useState(false);

    // Limit visible sources in stack
    const MAX_VISIBLE = 3;
    const visibleSources = context.slice(0, MAX_VISIBLE);
    const overflowCount = Math.max(0, context.length - MAX_VISIBLE);

    return (
        <div className="news-component-wrapper" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', marginTop: '1rem' }}>
            <div className="news-sources-stack" id="newsSourcesStack">
                <span className="news-sources-label">SOURCES:</span>
                <div className="news-sources-container">
                    {visibleSources.map((source, i) => {
                        const domain = getDomain(source.link);
                        const favicon = domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=32` : null;

                        return (
                            <div
                                key={i}
                                className={`source-icon ${source.is_trusted ? 'is-trusted' : ''}`}
                                style={{ zIndex: MAX_VISIBLE - i }}
                                title={source.headline || source.title || source.source}
                                data-source={source.source}
                            >
                                {favicon ? (
                                    <img src={favicon} alt="" onError={(e) => {
                                        e.target.style.display = 'none';
                                        e.target.parentElement.innerText = (source.source || 'UN').substring(0, 2).toUpperCase();
                                    }} />
                                ) : (
                                    (source.source || 'UN').substring(0, 2).toUpperCase()
                                )}
                            </div>
                        );
                    })}

                    {overflowCount > 0 && (
                        <div
                            className="source-icon source-icon-more"
                            style={{ zIndex: 0 }}
                            onClick={() => setExpanded(!expanded)}
                            title="View all sources"
                        >
                            +{overflowCount}
                        </div>
                    )}
                </div>
            </div>

            {/* Expanded List */}
            {expanded && (
                <div className="news-sources-expanded" id="newsSourcesExpanded">
                    <div className="news-sources-expanded-header">
                        <span>ALL SOURCES ({context.length})</span>
                        <span className="news-sources-close" onClick={() => setExpanded(false)}>×</span>
                    </div>
                    {context.map((source, i) => {
                        const domain = getDomain(source.link);
                        const favicon = domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=32` : null;
                        return (
                            <div key={i} className="news-source-item">
                                <span className="source-index">[{i + 1}]</span>
                                {favicon && <img className="source-favicon" src={favicon} alt="" />}
                                <div>
                                    <a href={source.link || '#'} target="_blank" rel="noopener noreferrer">{source.headline || source.title || 'Untitled'}</a>
                                    <div className="source-name">{source.source || 'Unknown Source'}</div>
                                </div>
                            </div>
                        );
                    })}
                    <style jsx="true">{`
                        .source-index {
                            font-family: monospace;
                            color: #4facfe;
                            margin-right: 8px;
                            font-size: 0.9rem;
                            min-width: 25px;
                        }
                    `}</style>
                </div>
            )}
        </div>
    );
};

NewsSources.propTypes = {
    context: PropTypes.array
};

export default NewsSources;
