import PropTypes from 'prop-types';

const WhatCouldGoWrong = ({ risks }) => {
    // 'risks' passed here should be the specifics from AI analysis
    // app.js populates #wcgwList from a subset of risks or specific field?
    // app.js line 923 checks ai.risks

    if (!risks || risks.length === 0) return null;

    // Filter for "severe" or "critical" keywords, or just take the top 3 risks to separate them from the main list?
    // app.js used: "aiRisks" for the main list, and "wcgwList" for... wait.
    // IN app.js, `updateAISection` maps `ai.risks` to `#aiRisks`.
    // It DOES NOT map to `#wcgwList` automatically!
    // I must have missed where `#wcgwList` is populated in `app.js`.
    // Searching app.js for `wcgwList`... it is ONLY in the `elements` definition?
    // Wait, checking `app.js` content snippet I read...
    // I don't see `wcgwList` being populated in the provided `app.js` lines 800+.
    // It might be dead code or populated elsewhere.

    // HOWEVER, the user said "missing features". WCGW is visually distinct.
    // I will use the AI `risks` array to populate this if it's not empty,
    // acting as a "Key Downside Risks" section.

    return (
        <div id="wcgwSection" className="wcgw-section">
            <div className="wcgw-header">WHAT COULD GO WRONG</div>
            <ul id="wcgwList" className="wcgw-list">
                {risks.slice(0, 3).map((risk, idx) => {
                    let text = typeof risk === 'object' ? (risk.factor || risk.description || JSON.stringify(risk)) : risk;

                    // Clean "Title: Description" format
                    if (text && typeof text === 'string' && text.includes(':')) {
                        text = text.substring(text.indexOf(':') + 1).trim();
                    }

                    return <li key={idx} className="wcgw-item">{text}</li>;
                })}
            </ul>
        </div>
    );
};

WhatCouldGoWrong.propTypes = {
    risks: PropTypes.array
};

export default WhatCouldGoWrong;
